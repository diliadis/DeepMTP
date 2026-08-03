from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from DeepMTP import (
    DeepMTPConfig,
    FoundationCacheSpec,
    FoundationLoRASpec,
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
    ModelFactory,
    tokenize_foundation_sequences,
)
from DeepMTP.config import ConfigValidationError
from DeepMTP.data import SequenceInput, process_instance_features
from DeepMTP.data.sequences import collate_sequence_rows
from DeepMTP.integrations import huggingface
from DeepMTP.utils.utils import generate_config


class FakeTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    bos_token_id = None
    eos_token_id = None
    init_kwargs = {"_commit_hash": "tokenizer-commit"}

    def __init__(self) -> None:
        self.calls = []

    def __len__(self):
        return 8

    def __call__(self, values, **kwargs):
        self.calls.append((values, kwargs))
        rows = [[1, *([3] * len(value)), 2] for value in values]
        if kwargs["truncation"]:
            rows = [row[: kwargs["max_length"]] for row in rows]
        return {"input_ids": rows}


class FakeBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size=8,
        hidden_size=4,
        padding_idx=0,
        revision="model-commit",
    ):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocabulary_size,
            pad_token_id=padding_idx,
            _commit_hash=revision,
        )
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        with torch.no_grad():
            for token_id in range(vocabulary_size):
                self.embedding.weight[token_id] = torch.arange(
                    token_id * 10,
                    token_id * 10 + hidden_size,
                    dtype=torch.float32,
                )
        self.forward_training_modes = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids, attention_mask):
        self.forward_training_modes.append(self.training)
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=hidden[:, 0] + 100,
        )


class FakePeftModel(nn.Module):
    def __init__(self, base_model, peft_config):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.peft_config = peft_config
        self.lora_adapter = nn.Parameter(torch.tensor(0.0))

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def forward(self, *, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return SimpleNamespace(
            last_hidden_state=outputs.last_hidden_state + self.lora_adapter,
            pooler_output=outputs.pooler_output + self.lora_adapter,
        )


class FakePeft:
    TaskType = SimpleNamespace(FEATURE_EXTRACTION="FEATURE_EXTRACTION")

    def __init__(self):
        self.configs = []

    def LoraConfig(self, **values):
        config = SimpleNamespace(**values)
        self.configs.append(config)
        return config

    @staticmethod
    def get_peft_model(model, config):
        return FakePeftModel(model, config)


def make_spec(**overrides):
    values = {
        "model_name": "example/foundation-model",
        "revision": "requested-model-revision",
        "tokenizer_revision": "requested-tokenizer-revision",
        "max_length": 8,
    }
    values.update(overrides)
    return FoundationModelSpec(**values)


def make_lora_spec(**overrides):
    values = {
        "rank": 4,
        "alpha": 8,
        "dropout": 0.1,
        "target_modules": ["embedding"],
    }
    values.update(overrides)
    return FoundationLoRASpec(**values)


def make_config(**overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "dot_product",
        "compute_mode": "cpu",
        "metrics": ["accuracy"],
        "metrics_average": ["macro"],
        "instance_branch_architecture": "SEQUENCE",
        "instance_branch_input_dim": 8,
        "instance_branch_sequence_encoder": "foundation",
        "instance_branch_sequence_output_dim": 3,
        "instance_branch_sequence_padding_idx": 0,
        "instance_branch_foundation_model": make_spec().to_dict(),
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
        "target_branch_nodes_per_layer": [3],
        "embedding_size": 3,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def make_sequence_input():
    return SequenceInput(
        token_ids=torch.tensor([[1, 2, 0], [3, 0, 0]]),
        attention_mask=torch.tensor([[True, True, False], [True, False, False]]),
        lengths=torch.tensor([2, 1]),
        padding_idx=0,
    )


def test_foundation_spec_normalizes_and_round_trips():
    spec = make_spec(pooling="CLS", train_mode="FULL")

    restored = FoundationModelSpec.from_config(spec.to_dict())

    assert restored.pooling == "cls"
    assert restored.train_mode == "full"
    assert restored.tokenizer_name == restored.model_name
    assert restored.tokenizer_revision == "requested-tokenizer-revision"


def test_foundation_spec_preserves_legacy_positional_argument_order():
    spec = FoundationModelSpec(
        "example/model",
        "model-revision",
        "example/tokenizer",
        "tokenizer-revision",
        "CLS",
        "FULL",
        True,
        True,
        8,
        True,
        False,
        (1, 2),
        "resolved-model",
        "resolved-tokenizer",
    )

    assert spec.train_mode == "full"
    assert spec.trust_remote_code is True
    assert spec.local_files_only is True
    assert spec.max_length == 8
    assert spec.truncation is True
    assert spec.exclude_special_tokens_from_pooling is False
    assert spec.pooling_excluded_token_ids == (1, 2)
    assert spec.resolved_model_revision == "resolved-model"
    assert spec.resolved_tokenizer_revision == "resolved-tokenizer"
    assert spec.lora is None


def test_lora_spec_normalizes_and_round_trips():
    spec = make_spec(
        train_mode="LORA",
        lora={
            "rank": 4,
            "alpha": 8,
            "dropout": 0,
            "target_modules": [" query ", "value"],
        },
    )

    restored = FoundationModelSpec.from_config(spec.to_dict())

    assert restored.train_mode == "lora"
    assert isinstance(restored.lora, FoundationLoRASpec)
    assert restored.lora.dropout == 0.0
    assert restored.lora.target_modules == ("query", "value")
    assert restored.to_dict()["lora"] == {
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["query", "value"],
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rank": 0}, "rank"),
        ({"alpha": True}, "alpha"),
        ({"dropout": 1.0}, "dropout"),
        ({"target_modules": []}, "at least one"),
        ({"target_modules": "query"}, "list or tuple"),
        ({"target_modules": ["query", "query"]}, "duplicates"),
    ],
)
def test_lora_spec_rejects_invalid_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_lora_spec(**overrides)


def test_foundation_spec_requires_lora_mode_and_settings_together():
    with pytest.raises(ValueError, match="requires explicit lora"):
        make_spec(train_mode="lora")
    with pytest.raises(ValueError, match="require train_mode='lora'"):
        make_spec(lora=make_lora_spec())
    with pytest.raises(ValueError, match="unknown settings"):
        make_spec(train_mode="lora", lora={"target_modules": ["query"], "r": 4})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_name": ""}, "model_name"),
        ({"pooling": "maximum"}, "pooling"),
        ({"train_mode": "adapter"}, "train_mode"),
        ({"trust_remote_code": 1}, "boolean"),
        ({"exclude_special_tokens_from_pooling": 1}, "boolean"),
        ({"pooling_excluded_token_ids": [1, -1]}, "non-negative"),
        ({"pooling_excluded_token_ids": [1, 1]}, "duplicates"),
        ({"max_length": 0}, "max_length"),
        ({"truncation": True, "max_length": None}, "requires"),
    ],
)
def test_foundation_spec_rejects_invalid_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_spec(**overrides)


def test_tokenizer_returns_unpadded_rows_and_resolved_metadata(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_tokenizer",
        lambda _spec: tokenizer,
    )

    tokenized = tokenize_foundation_sequences(
        ["AB", "C"],
        make_spec(pooling_excluded_token_ids=[7]),
    )
    processed = process_instance_features(tokenized, feature_kind="sequence")

    assert tokenized.vocabulary_size == 8
    assert tokenized.padding_idx == 0
    assert tokenized.spec.resolved_tokenizer_revision == "tokenizer-commit"
    assert tokenized.spec.pooling_excluded_token_ids == (7, 1, 2)
    assert [row.tolist() for row in tokenized] == [[1, 3, 3, 2], [1, 3, 2]]
    assert tokenizer.calls[0][1]["padding"] is False
    assert tokenizer.calls[0][1]["truncation"] is False
    assert processed["info"] == "sequence"


def test_tokenizer_requires_explicit_truncation_for_long_rows(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_tokenizer",
        lambda _spec: FakeTokenizer(),
    )

    with pytest.raises(ValueError, match="Enable truncation explicitly"):
        tokenize_foundation_sequences(
            ["TOO-LONG"],
            make_spec(max_length=4),
        )

    tokenized = tokenize_foundation_sequences(
        ["TOO-LONG"],
        make_spec(max_length=4, truncation=True),
    )

    assert len(tokenized[0]) == 4


def test_missing_transformers_dependency_has_install_guidance(monkeypatch):
    def missing_import(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(huggingface, "import_module", missing_import)

    with pytest.raises(ModuleNotFoundError, match=r"DeepMTP\[foundation\]"):
        huggingface._transformers_module()


def test_missing_peft_dependency_has_install_guidance(monkeypatch):
    def missing_import(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(huggingface, "import_module", missing_import)

    with pytest.raises(ModuleNotFoundError, match=r"DeepMTP\[foundation\]"):
        huggingface._peft_module()


@pytest.mark.parametrize("pooling", ["cls", "masked_mean", "pooler"])
def test_foundation_encoder_pools_and_projects(pooling, monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(pooling=pooling),
        vocabulary_size=8,
        output_dim=4,
        padding_idx=0,
    )
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(4))
        encoder.projection.bias.zero_()

    output = encoder(make_sequence_input())

    if pooling == "cls":
        expected = torch.tensor([[10.0, 11.0, 12.0, 13.0], [30, 31, 32, 33]])
    elif pooling == "masked_mean":
        expected = torch.tensor([[15.0, 16.0, 17.0, 18.0], [30, 31, 32, 33]])
    else:
        expected = torch.tensor([[110.0, 111.0, 112.0, 113.0], [130, 131, 132, 133]])
    assert torch.equal(output, expected)


def test_masked_mean_excludes_tokenizer_boundary_tokens(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    values = SequenceInput(
        token_ids=torch.tensor([[1, 3, 2], [1, 4, 2]]),
        attention_mask=torch.ones((2, 3), dtype=torch.bool),
        lengths=torch.tensor([3, 3]),
        padding_idx=0,
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(pooling_excluded_token_ids=[1, 2]),
        vocabulary_size=8,
        output_dim=4,
        padding_idx=0,
    )
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(4))
        encoder.projection.bias.zero_()

    output = encoder(values)

    assert torch.equal(
        output,
        torch.tensor([[30.0, 31.0, 32.0, 33.0], [40.0, 41.0, 42.0, 43.0]]),
    )


def test_masked_mean_special_token_exclusion_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    values = SequenceInput(
        token_ids=torch.tensor([[1, 3, 2]]),
        attention_mask=torch.ones((1, 3), dtype=torch.bool),
        lengths=torch.tensor([3]),
        padding_idx=0,
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(
            pooling_excluded_token_ids=[1, 2],
            exclude_special_tokens_from_pooling=False,
        ),
        vocabulary_size=8,
        output_dim=4,
        padding_idx=0,
    )
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(4))
        encoder.projection.bias.zero_()

    output = encoder(values)

    assert torch.equal(output, torch.tensor([[20.0, 21.0, 22.0, 23.0]]))


def test_masked_mean_rejects_rows_containing_only_special_tokens(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    values = SequenceInput(
        token_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        lengths=torch.tensor([2]),
        padding_idx=0,
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(pooling_excluded_token_ids=[1, 2]),
        vocabulary_size=8,
        output_dim=4,
        padding_idx=0,
    )

    with pytest.raises(ValueError, match="no non-special token"):
        encoder(values)


def test_frozen_foundation_backbone_stays_in_eval_and_projection_trains(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(train_mode="frozen"),
        vocabulary_size=8,
        output_dim=3,
        padding_idx=0,
    )

    encoder.train()
    encoder(make_sequence_input()).sum().backward()

    assert encoder.training
    assert not encoder.backbone.training
    assert encoder.backbone.forward_training_modes == [False]
    assert encoder.backbone.embedding.weight.grad is None
    assert isinstance(encoder.projection, nn.Linear)
    assert encoder.projection.weight.grad is not None


def test_equal_width_frozen_backbone_still_has_a_trainable_projection(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(train_mode="frozen"),
        vocabulary_size=8,
        output_dim=4,
        padding_idx=0,
    )

    encoder(make_sequence_input()).sum().backward()

    assert isinstance(encoder.projection, nn.Linear)
    assert encoder.projection.weight.grad is not None


def test_full_foundation_backbone_receives_gradients(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(train_mode="full"),
        vocabulary_size=8,
        output_dim=3,
        padding_idx=0,
    )

    encoder.train()
    encoder(make_sequence_input()).sum().backward()

    assert encoder.backbone.training
    assert encoder.backbone.embedding.weight.grad is not None


def test_lora_foundation_backbone_trains_only_adapters_and_projection(monkeypatch):
    fake_peft = FakePeft()
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    monkeypatch.setattr(huggingface, "_peft_module", lambda: fake_peft)
    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(train_mode="lora", lora=make_lora_spec()),
        vocabulary_size=8,
        output_dim=3,
        padding_idx=0,
    )

    encoder.train()
    encoder(make_sequence_input()).sum().backward()

    assert encoder.backbone.training
    assert encoder.backbone.base_model.embedding.weight.grad is None
    assert encoder.backbone.lora_adapter.grad is not None
    assert encoder.projection.weight.grad is not None
    assert fake_peft.configs[0].task_type == "FEATURE_EXTRACTION"
    assert fake_peft.configs[0].r == 4
    assert fake_peft.configs[0].lora_alpha == 8
    assert fake_peft.configs[0].lora_dropout == 0.1
    assert fake_peft.configs[0].target_modules == ["embedding"]
    assert fake_peft.configs[0].bias == "none"


def test_lora_foundation_rejects_unknown_target_modules(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )

    with pytest.raises(ValueError, match="Available leaf-module suffixes"):
        HuggingFaceSequenceEncoder(
            spec=make_spec(
                train_mode="lora",
                lora=make_lora_spec(target_modules=["does_not_exist"]),
            ),
            vocabulary_size=8,
            output_dim=3,
            padding_idx=0,
        )


def test_lora_foundation_state_dict_restores_adapter_and_projection(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    monkeypatch.setattr(huggingface, "_peft_module", lambda: FakePeft())
    arguments = {
        "spec": make_spec(train_mode="lora", lora=make_lora_spec()),
        "vocabulary_size": 8,
        "output_dim": 3,
        "padding_idx": 0,
    }
    first = HuggingFaceSequenceEncoder(**arguments)
    with torch.no_grad():
        first.backbone.lora_adapter.fill_(2.0)
    expected = first(make_sequence_input())
    state = deepcopy(first.state_dict())

    second = HuggingFaceSequenceEncoder(**arguments)
    with torch.no_grad():
        second.projection.weight.zero_()
    second.load_state_dict(state)

    assert any("lora_adapter" in name for name in state)
    assert torch.equal(second(make_sequence_input()), expected)


def test_foundation_encoder_state_dict_restores_backbone_and_projection(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    first = HuggingFaceSequenceEncoder(
        spec=make_spec(),
        vocabulary_size=8,
        output_dim=3,
        padding_idx=0,
    )
    expected = first(make_sequence_input())
    state = deepcopy(first.state_dict())

    second = HuggingFaceSequenceEncoder(
        spec=make_spec(),
        vocabulary_size=8,
        output_dim=3,
        padding_idx=0,
    )
    with torch.no_grad():
        second.projection.weight.zero_()
    second.load_state_dict(state)

    assert any(name.startswith("backbone.") for name in state)
    assert torch.equal(second(make_sequence_input()), expected)


def test_config_validates_foundation_settings_and_legacy_defaults():
    config = make_config(instance_branch_sequence_encoder="FOUNDATION")
    legacy = make_config(
        instance_branch_sequence_encoder="gru",
        instance_branch_foundation_model=None,
    ).to_dict(include_defaults=True)
    legacy.pop("instance_branch_foundation_model")
    legacy.pop("target_branch_foundation_model")

    restored = DeepMTPConfig.from_mapping(legacy)

    assert config.instance_branch_sequence_encoder == "foundation"
    assert (
        config.instance_branch_foundation_model["model_name"]
        == "example/foundation-model"
    )
    assert restored.instance_branch_foundation_model is None
    assert restored.target_branch_foundation_model is None

    with pytest.raises(ConfigValidationError, match="foundation_model"):
        make_config(instance_branch_foundation_model=None)


def test_generate_config_forwards_foundation_sequence_settings():
    config = generate_config(
        validation_setting="B",
        problem_mode="classification",
        general_architecture_version="dot_product",
        compute_mode="cpu",
        metrics=["accuracy"],
        metrics_average=["macro"],
        instance_branch_architecture="SEQUENCE",
        instance_branch_input_dim=8,
        instance_branch_params={
            "instance_branch_sequence_encoder": "foundation",
            "instance_branch_sequence_padding_idx": 0,
            "instance_branch_foundation_model": make_spec().to_dict(),
        },
        target_branch_architecture="MLP",
        target_branch_input_dim=2,
        target_branch_params={"target_branch_nodes_per_layer": [3]},
        embedding_size=3,
    )

    assert config["instance_branch_sequence_encoder"] == "foundation"
    assert (
        config["instance_branch_foundation_model"]["model_name"]
        == "example/foundation-model"
    )


def test_model_factory_builds_foundation_branch_and_records_revision(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(revision="resolved-model-revision"),
    )
    config = make_config()

    bundle = ModelFactory(config).build()
    output = bundle.model(make_sequence_input(), torch.ones(2, 2))

    assert isinstance(bundle.instance_branch, HuggingFaceSequenceEncoder)
    assert output.shape == (2, 1)
    assert (
        config.instance_branch_foundation_model["resolved_model_revision"]
        == "resolved-model-revision"
    )


@pytest.mark.parametrize("train_mode", ["frozen", "lora"])
def test_composite_branch_supports_a_foundation_component(monkeypatch, train_mode):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(),
    )
    monkeypatch.setattr(huggingface, "_peft_module", lambda: FakePeft())
    foundation_spec = make_spec(
        train_mode=train_mode,
        lora=make_lora_spec() if train_mode == "lora" else None,
    )
    config = make_config(
        general_architecture_version="mlp",
        instance_branch_architecture="COMPOSITE",
        instance_branch_composite_components={
            "descriptors": {
                "architecture": "MLP",
                "input_dim": 2,
                "output_dim": 2,
                "nodes_per_layer": [2],
            },
            "tokens": {
                "architecture": "SEQUENCE",
                "input_dim": 8,
                "output_dim": 3,
                "encoder": "foundation",
                "padding_idx": 0,
                "foundation_model": foundation_spec.to_dict(),
            },
        },
        comb_mlp_nodes_per_layer=[3],
    )

    bundle = ModelFactory(config).build()
    component = bundle.instance_branch.component_encoders["tokens"]

    assert isinstance(component, HuggingFaceSequenceEncoder)
    if train_mode == "lora":
        assert isinstance(component.backbone, FakePeftModel)
    assert (
        config.instance_branch_composite_components["tokens"]["foundation_model"][
            "resolved_model_revision"
        ]
        == "model-commit"
    )


def test_composite_branch_reuses_cached_foundation_outputs(monkeypatch):
    backbone = FakeBackbone()
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: backbone,
    )
    config = make_config(
        general_architecture_version="mlp",
        instance_branch_architecture="COMPOSITE",
        instance_branch_composite_components={
            "descriptors": {
                "architecture": "MLP",
                "input_dim": 2,
                "output_dim": 2,
                "nodes_per_layer": [2],
            },
            "tokens": {
                "architecture": "SEQUENCE",
                "input_dim": 8,
                "output_dim": 3,
                "encoder": "foundation",
                "padding_idx": 0,
                "foundation_model": make_spec(
                    cache=FoundationCacheSpec(memory_capacity=8)
                ).to_dict(),
            },
        },
        comb_mlp_nodes_per_layer=[3],
    )

    bundle = ModelFactory(config).build()
    component = bundle.instance_branch.component_encoders["tokens"]
    component(make_sequence_input())
    component(make_sequence_input())

    assert isinstance(component, HuggingFaceSequenceEncoder)
    assert backbone.forward_training_modes == [False]
    assert component.cache_stats["memory_hits"] == 2


def test_foundation_encoder_accepts_unused_model_vocabulary_capacity(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(vocabulary_size=9),
    )

    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(),
        vocabulary_size=8,
        output_dim=3,
        padding_idx=0,
    )

    assert encoder.vocabulary_size == 8


def test_foundation_encoder_accepts_model_padding_metadata_mismatch(monkeypatch):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(padding_idx=1),
    )

    encoder = HuggingFaceSequenceEncoder(
        spec=make_spec(),
        vocabulary_size=8,
        output_dim=3,
        padding_idx=0,
    )

    assert encoder.padding_idx == 0
    assert encoder.backbone.config.pad_token_id == 1


def test_foundation_encoder_rejects_insufficient_model_vocabulary_capacity(
    monkeypatch,
):
    monkeypatch.setattr(
        huggingface,
        "_load_huggingface_model",
        lambda _spec: FakeBackbone(vocabulary_size=7),
    )

    with pytest.raises(ValueError, match="capacity is smaller"):
        HuggingFaceSequenceEncoder(
            spec=make_spec(),
            vocabulary_size=8,
            output_dim=3,
            padding_idx=0,
        )


@pytest.mark.integration
@pytest.mark.parametrize("train_mode", ["frozen", "full", "lora"])
def test_foundation_encoder_loads_a_local_transformers_model(tmp_path, train_mode):
    transformers = pytest.importorskip("transformers")
    if train_mode == "lora":
        pytest.importorskip("peft")
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
    vocabulary_path = tmp_path / "vocab.txt"
    vocabulary_path.write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = transformers.BertTokenizer(str(vocabulary_path))
    tokenizer.save_pretrained(tmp_path)
    model = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=8,
            pad_token_id=0,
        )
    )
    model.save_pretrained(tmp_path)
    spec = FoundationModelSpec(
        model_name=str(tmp_path),
        local_files_only=True,
        max_length=8,
        train_mode=train_mode,
        lora=(
            FoundationLoRASpec(
                rank=2,
                alpha=4,
                target_modules=("query", "value"),
            )
            if train_mode == "lora"
            else None
        ),
    )
    tokenized = tokenize_foundation_sequences(["hello world", "deepmtp"], spec)
    token_ids, attention_mask, lengths = collate_sequence_rows(
        tokenized,
        padding_idx=tokenized.padding_idx,
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=tokenized.spec,
        vocabulary_size=tokenized.vocabulary_size,
        output_dim=3,
        padding_idx=tokenized.padding_idx,
    )

    values = SequenceInput(
        token_ids=token_ids,
        attention_mask=attention_mask,
        lengths=lengths,
        padding_idx=tokenized.padding_idx,
    )
    output = encoder(values)

    assert output.shape == (2, 3)
    if train_mode == "lora":
        trainable_names = {
            name
            for name, parameter in encoder.backbone.named_parameters()
            if parameter.requires_grad
        }
        assert trainable_names
        assert all("lora_" in name for name in trainable_names)
        state = deepcopy(encoder.state_dict())
        restored = HuggingFaceSequenceEncoder(
            spec=tokenized.spec,
            vocabulary_size=tokenized.vocabulary_size,
            output_dim=3,
            padding_idx=tokenized.padding_idx,
        )
        restored.load_state_dict(state)
        encoder.eval()
        restored.eval()
        assert torch.allclose(restored(values), encoder(values))
