import pytest
import torch
from torch import nn

import DeepMTP.models.factory as model_factory
from DeepMTP import (
    DeepMTPConfig,
    IDEmbedding,
    ModelFactory,
    TwoBranchDotProductModel,
    TwoBranchKroneckerModel,
    TwoBranchMLPModel,
)


def make_config(**overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "dot_product",
        "compute_mode": "cpu",
        "metrics": ["accuracy"],
        "metrics_average": ["macro"],
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 2,
        "instance_branch_nodes_per_layer": [4],
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 3,
        "target_branch_nodes_per_layer": [5],
        "embedding_size": 3,
        "random_seed": 17,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


@pytest.mark.parametrize(
    ("architecture", "model_type"),
    [
        ("dot_product", TwoBranchDotProductModel),
        ("mlp", TwoBranchMLPModel),
        ("kronecker", TwoBranchKroneckerModel),
    ],
)
def test_model_factory_builds_each_combination_architecture(
    architecture,
    model_type,
):
    bundle = ModelFactory(
        make_config(general_architecture_version=architecture)
    ).build()

    output = bundle.model(
        torch.ones(2, 2),
        torch.ones(2, 3),
    )

    assert isinstance(bundle.model, model_type)
    assert output.shape == (2, 1)


@pytest.mark.parametrize("architecture", ["dot_product", "mlp", "kronecker"])
def test_model_factory_builds_id_embeddings_for_every_combination(architecture):
    bundle = ModelFactory(
        make_config(
            validation_setting="A",
            metrics_average=["micro"],
            general_architecture_version=architecture,
            instance_branch_architecture="EMBEDDING",
            instance_branch_input_dim=4,
            target_branch_architecture="EMBEDDING",
            target_branch_input_dim=5,
            embedding_size=3,
        )
    ).build()

    output = bundle.model(
        torch.tensor([0, 3], dtype=torch.int64),
        torch.tensor([4, 1], dtype=torch.int64),
    )

    assert isinstance(bundle.instance_branch, IDEmbedding)
    assert isinstance(bundle.target_branch, IDEmbedding)
    assert bundle.instance_branch.embedding.weight.shape == (4, 3)
    assert bundle.target_branch.embedding.weight.shape == (5, 3)
    assert output.shape == (2, 1)


def test_model_factory_supports_features_with_known_target_id_embeddings():
    bundle = ModelFactory(
        make_config(
            general_architecture_version="mlp",
            target_branch_architecture="EMBEDDING",
            target_branch_input_dim=5,
            embedding_size=3,
        )
    ).build()

    output = bundle.model(
        torch.ones(2, 2),
        torch.tensor([4, 1], dtype=torch.int64),
    )

    assert isinstance(bundle.target_branch, IDEmbedding)
    assert output.shape == (2, 1)


def test_built_in_models_do_not_read_legacy_config_snapshot():
    factory = ModelFactory(make_config(general_architecture_version="mlp"))
    factory.legacy_config["comb_mlp_nodes_per_layer"] = [-1]
    factory.legacy_config["dropout_rate"] = 2.0

    bundle = factory.build()

    assert bundle.model(torch.ones(2, 2), torch.ones(2, 3)).shape == (2, 1)


def test_combined_mlp_builds_from_repeated_integer_width():
    bundle = ModelFactory(
        make_config(
            general_architecture_version="mlp",
            comb_mlp_nodes_per_layer=6,
            comb_mlp_layers=2,
        )
    ).build()

    assert bundle.model(torch.ones(2, 2), torch.ones(2, 3)).shape == (2, 1)


def test_seeded_model_creation_is_repeatable_and_restores_global_rng():
    config = make_config(random_seed=23)
    global_state = torch.random.get_rng_state()

    first = ModelFactory(config).build()
    state_after_first_build = torch.random.get_rng_state()
    second = ModelFactory(config).build()

    assert torch.equal(global_state, state_after_first_build)
    for first_parameter, second_parameter in zip(
        first.model.parameters(),
        second.model.parameters(),
        strict=True,
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_different_model_seeds_produce_different_parameters():
    first = ModelFactory(make_config(random_seed=1)).build()
    second = ModelFactory(make_config(random_seed=2)).build()

    first_parameter = next(first.model.parameters())
    second_parameter = next(second.model.parameters())

    assert not torch.equal(first_parameter, second_parameter)


def test_model_factory_supports_custom_branch_factories():
    config = make_config(
        instance_branch_architecture="CUSTOM",
        target_branch_architecture="CUSTOM",
    )

    def instance_factory(legacy_config):
        return nn.Linear(2, legacy_config["embedding_size"])

    def target_factory(legacy_config):
        return nn.Linear(3, legacy_config["embedding_size"])

    bundle = ModelFactory(config).build(
        instance_branch_factory=instance_factory,
        target_branch_factory=target_factory,
    )

    assert bundle.model(torch.ones(2, 2), torch.ones(2, 3)).shape == (2, 1)
    assert bundle.instance_branch.input_kind == "custom"
    assert bundle.instance_branch.output_dim == 3


def test_non_dot_custom_branches_get_a_deprecated_compatibility_contract():
    config = make_config(
        general_architecture_version="mlp",
        instance_branch_architecture="CUSTOM",
        target_branch_architecture="CUSTOM",
    )

    with pytest.warns(FutureWarning, match="Declare output_dim explicitly"):
        bundle = ModelFactory(config).build(
            instance_branch_factory=lambda _config: nn.Linear(2, 6),
            target_branch_factory=lambda _config: nn.Linear(3, 7),
        )

    assert bundle.instance_branch.output_dim == 6
    assert bundle.target_branch.output_dim == 7
    assert bundle.model(torch.ones(2, 2), torch.ones(2, 3)).shape == (2, 1)


def test_encoder_contract_metadata_preserves_checkpoint_parameter_keys():
    config = make_config(general_architecture_version="mlp")
    original = ModelFactory(config).build()
    state_dict = original.model.state_dict()
    restored = ModelFactory(config).build()

    restored.model.load_state_dict(state_dict, strict=True)

    assert state_dict
    assert all("input_kind" not in key for key in state_dict)
    assert all("output_dim" not in key for key in state_dict)


def test_custom_factory_rejects_an_input_kind_incompatible_with_configured_data():
    class IDOnlyEncoder(nn.Linear):
        input_kind = "id"
        output_dim = 3

    with pytest.raises(ValueError, match="MLP data requires 'dense'"):
        ModelFactory(make_config()).build(
            instance_branch_factory=lambda _config: IDOnlyEncoder(2, 3),
        )


def test_custom_architecture_requires_a_factory():
    config = make_config(instance_branch_architecture="CUSTOM")

    with pytest.raises(ValueError, match="instance_branch_factory"):
        ModelFactory(config).build()


def test_custom_factory_must_return_a_torch_module():
    config = make_config(instance_branch_architecture="CUSTOM")

    with pytest.raises(TypeError, match="torch.nn.Module"):
        ModelFactory(config).build(
            instance_branch_factory=lambda legacy_config: object()
        )


def test_model_factory_propagates_offline_convolutional_weights_setting(monkeypatch):
    captured = {}

    def fake_convnet(*args, pretrained):
        captured["pretrained"] = pretrained
        return nn.Identity()

    monkeypatch.setattr(model_factory, "ConvNet", fake_convnet)
    factory = ModelFactory(
        make_config(
            instance_branch_architecture="CONV",
            instance_branch_conv_pretrained=False,
        )
    )

    factory._build_branch("instance", output_dim=3, custom_factory=None)

    assert captured == {"pretrained": False}
