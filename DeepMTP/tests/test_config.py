import pytest

from DeepMTP import __version__
from DeepMTP.config import (
    ConfigDeprecationWarning,
    ConfigValidationError,
    DeepMTPConfig,
)
from DeepMTP.utils.utils import generate_config


def make_valid_config(**overrides):
    arguments = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "dot_product",
        "compute_mode": "cpu",
        "metrics": ["accuracy"],
        "metrics_average": ["macro"],
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 3,
        "instance_branch_params": {
            "instance_branch_nodes_per_layer": [4],
            "instance_branch_layers": None,
            "instance_branch_nodes_reducing_factor": 2,
        },
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 2,
        "target_branch_params": {
            "target_branch_nodes_per_layer": [4],
            "target_branch_layers": None,
            "target_branch_nodes_reducing_factor": 2,
        },
    }
    arguments.update(overrides)
    return generate_config(**arguments)


def test_typed_config_round_trip_preserves_extension_metadata():
    config = make_valid_config()
    config["budget"] = 12

    typed_config = DeepMTPConfig.from_mapping(config)

    assert typed_config.validation_setting == "B"
    assert typed_config.problem_mode == "classification"
    assert typed_config.to_dict() == config


def test_runtime_dict_materializes_defaults_for_partial_mapping():
    typed_config = DeepMTPConfig.from_mapping(
        {
            "validation_setting": "B",
            "problem_mode": "classification",
            "instance_branch_architecture": "MLP",
            "instance_branch_input_dim": 3,
            "target_branch_architecture": "MLP",
            "target_branch_input_dim": 2,
        }
    )

    assert "verbose" not in typed_config.to_dict()
    assert typed_config.to_dict(include_defaults=True)["verbose"] is False


def test_typed_config_does_not_copy_opaque_transform_objects():
    class NonCopyableTransform:
        def __deepcopy__(self, memo):
            raise AssertionError("opaque integrations must not be deep-copied")

    transform = NonCopyableTransform()
    config = make_valid_config(instance_train_transforms=transform)

    typed_config = DeepMTPConfig.from_mapping(config)

    assert typed_config.instance_train_transforms is transform
    assert typed_config.to_dict()["instance_train_transforms"] is transform


def test_typed_config_normalizes_public_names():
    config = make_valid_config()
    config.update(
        {
            "validation_setting": "b",
            "problem_mode": "CLASSIFICATION",
            "general_architecture_version": "DOT_PRODUCT",
            "instance_branch_architecture": "mlp",
            "target_branch_architecture": "mlp",
            "metrics": ["ACCURACY"],
            "metrics_average": ["MACRO"],
        }
    )

    normalized = DeepMTPConfig.from_mapping(config)

    assert normalized.validation_setting == "B"
    assert normalized.problem_mode == "classification"
    assert normalized.general_architecture_version == "dot_product"
    assert normalized.instance_branch_architecture == "MLP"
    assert normalized.metrics == ["accuracy"]
    assert normalized.metrics_average == ["macro"]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("validation_setting", "Z"),
        ("problem_mode", "forecasting"),
        ("general_architecture_version", "transformer"),
        ("compute_mode", "cuda:all"),
        ("learning_rate", 0),
        ("dropout_rate", 1),
        ("num_workers", -1),
        ("train_batchsize", 0),
        ("random_seed", -1),
        ("eval_every_n_epochs", 0),
        ("instance_branch_input_dim", 0),
        ("metric_to_optimize_best_epoch_selection", "accuracy_micro"),
        ("wandb_project_name", "project-without-entity"),
    ],
)
def test_typed_config_rejects_invalid_core_values(field_name, invalid_value):
    config = make_valid_config()
    config[field_name] = invalid_value

    with pytest.raises(ConfigValidationError, match=field_name):
        DeepMTPConfig.from_mapping(config)


def test_typed_config_rejects_non_micro_averaging_for_setting_a():
    config = make_valid_config()
    config["validation_setting"] = "A"
    config["metrics_average"] = ["macro", "micro"]

    with pytest.raises(
        ConfigValidationError,
        match=r"metrics_average must be \['micro'\]",
    ):
        DeepMTPConfig.from_mapping(config)


def test_generate_config_mutable_defaults_are_isolated():
    first = make_valid_config()
    first["metrics"].append("precision")
    first["instance_branch_nodes_per_layer"].append(99)

    second = make_valid_config()

    assert second["metrics"] == ["accuracy"]
    assert second["instance_branch_nodes_per_layer"] == [4]


def test_inert_legacy_options_emit_a_visible_deprecation_warning():
    config = make_valid_config()
    config["weighted_loss"] = True

    with pytest.warns(ConfigDeprecationWarning, match="weighted_loss"):
        DeepMTPConfig.from_mapping(config)


def test_package_exposes_a_version():
    assert isinstance(__version__, str)
    assert __version__
