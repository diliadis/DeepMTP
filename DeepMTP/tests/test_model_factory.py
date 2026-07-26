import pytest
import torch
from torch import nn

from DeepMTP import (
    DeepMTPConfig,
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
