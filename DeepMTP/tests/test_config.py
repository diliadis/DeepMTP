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
            "loss": "BINARY_CROSS_ENTROPY_WITH_LOGITS",
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
    assert normalized.loss == "binary_cross_entropy_with_logits"
    assert normalized.general_architecture_version == "dot_product"
    assert normalized.instance_branch_architecture == "MLP"
    assert normalized.metrics == ["accuracy"]
    assert normalized.metrics_average == ["macro"]


def test_sparse_branch_configuration_preserves_mlp_layer_options():
    config = make_valid_config(
        instance_branch_architecture="SPARSE",
        instance_branch_input_dim=100_000,
        instance_branch_params={
            "instance_branch_nodes_per_layer": [32, 8],
            "instance_branch_layers": None,
            "instance_branch_nodes_reducing_factor": 2,
        },
    )

    normalized = DeepMTPConfig.from_mapping(config)

    assert normalized.instance_branch_architecture == "SPARSE"
    assert normalized.instance_branch_input_dim == 100_000
    assert normalized.instance_branch_nodes_per_layer == [32, 8]


def test_sparse_branch_requires_a_positive_input_dimension():
    config = make_valid_config()
    config["instance_branch_architecture"] = "SPARSE"
    config["instance_branch_input_dim"] = None

    with pytest.raises(
        ConfigValidationError,
        match="instance_branch_input_dim",
    ):
        DeepMTPConfig.from_mapping(config)


@pytest.mark.parametrize(
    ("problem_mode", "metrics", "expected_loss"),
    [
        ("classification", ["accuracy"], "binary_cross_entropy_with_logits"),
        ("regression", ["MSE"], "mean_squared_error"),
    ],
)
def test_problem_modes_materialize_stable_default_losses(
    problem_mode,
    metrics,
    expected_loss,
):
    config = make_valid_config(problem_mode=problem_mode, metrics=metrics)

    assert config["loss"] == expected_loss
    assert DeepMTPConfig.from_mapping(config).loss == expected_loss


def test_multiclass_config_materializes_safe_defaults_and_output_width():
    config = make_valid_config(
        classification_mode="MULTICLASS",
        num_classes=4,
    )
    normalized = DeepMTPConfig.from_mapping(config)

    assert config["classification_mode"] == "multiclass"
    assert config["loss"] == "cross_entropy"
    assert config["multiclass_average"] == "macro"
    assert normalized.num_classes == 4
    assert normalized.output_dimension == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"classification_mode": "multiclass", "num_classes": None},
            "num_classes",
        ),
        (
            {"classification_mode": "multiclass", "num_classes": 2},
            "greater than or equal to 3",
        ),
        (
            {
                "classification_mode": "multiclass",
                "num_classes": 3,
                "loss": "binary_cross_entropy_with_logits",
            },
            "incompatible",
        ),
        (
            {
                "classification_mode": "multiclass",
                "num_classes": 3,
                "top_k": 2,
            },
            "top_k is not supported",
        ),
        (
            {
                "classification_mode": "multiclass",
                "num_classes": 3,
                "multiclass_average": "samples",
            },
            "multiclass_average",
        ),
        (
            {"classification_mode": "binary", "num_classes": 3},
            "num_classes is only valid",
        ),
        (
            {
                "problem_mode": "regression",
                "metrics": ["MSE"],
                "classification_mode": "binary",
            },
            "classification_mode must be None",
        ),
    ],
)
def test_multiclass_config_rejects_ambiguous_or_incompatible_options(
    overrides,
    message,
):
    with pytest.raises(ConfigValidationError, match=message):
        make_valid_config(**overrides)


@pytest.mark.parametrize("loss", ["mean_absolute_error", "huber"])
def test_regression_accepts_configurable_losses(loss):
    config = make_valid_config(
        problem_mode="regression",
        metrics=["MSE"],
        loss=loss.upper(),
    )

    assert config["loss"] == loss
    assert DeepMTPConfig.from_mapping(config).loss == loss


@pytest.mark.parametrize(
    ("problem_mode", "metrics", "loss"),
    [
        ("classification", ["accuracy"], "mean_squared_error"),
        ("regression", ["MSE"], "binary_cross_entropy_with_logits"),
        ("regression", ["MSE"], "unknown"),
    ],
)
def test_losses_must_be_compatible_with_problem_mode(
    problem_mode,
    metrics,
    loss,
):
    config = make_valid_config(problem_mode=problem_mode, metrics=metrics)
    config["loss"] = loss

    with pytest.raises(
        ConfigValidationError,
        match=rf"loss={loss!r} is incompatible",
    ):
        DeepMTPConfig.from_mapping(config)


def test_typed_config_normalizes_and_accepts_id_embedding_branches():
    config = make_valid_config(target_branch_input_dim=7, embedding_size=5)
    config["target_branch_architecture"] = "embedding"

    normalized = DeepMTPConfig.from_mapping(config)

    assert normalized.target_branch_architecture == "EMBEDDING"
    assert normalized.target_branch_input_dim == 7
    assert normalized.embedding_size == 5


def test_generate_config_accepts_and_normalizes_a_tabular_branch_schema():
    schema = {
        "numeric_columns": ["age"],
        "categorical_columns": {
            "site": {
                "categories": ["A", "B"],
                "embedding_dim": 2,
            }
        },
    }
    config = make_valid_config(
        instance_branch_architecture="TABULAR",
        instance_branch_input_dim=None,
        instance_branch_tabular_schema=schema,
        instance_branch_params={
            "instance_branch_nodes_per_layer": [6],
        },
    )

    normalized = DeepMTPConfig.from_mapping(config)

    assert normalized.instance_branch_architecture == "TABULAR"
    assert normalized.instance_branch_input_dim is None
    assert normalized.instance_branch_nodes_per_layer == [6]
    assert normalized.instance_branch_tabular_schema == {
        "numeric_columns": ["age"],
        "categorical_columns": {
            "site": {
                "categories": ["A", "B"],
                "embedding_dim": 2,
            }
        },
        "numeric_normalization": "standard",
        "numeric_missing": "mean",
        "categorical_missing": "unknown",
        "categorical_unknown": "unknown",
        "feature_gating": False,
    }


def test_tabular_branch_requires_a_valid_schema_and_matching_saved_state():
    config = make_valid_config()
    config["instance_branch_architecture"] = "TABULAR"
    config["instance_branch_tabular_schema"] = {
        "numeric_columns": ["age"],
        "categorical_columns": {},
    }
    config["instance_tabular_preprocessing_state"] = {
        "numeric_columns": ["dose"],
        "categorical_columns": [],
        "numeric_fill_values": [0.0],
        "numeric_offsets": [0.0],
        "numeric_scales": [1.0],
    }

    with pytest.raises(
        ConfigValidationError,
        match="numeric columns do not match",
    ):
        DeepMTPConfig.from_mapping(config)


@pytest.mark.parametrize(
    ("branch", "validation_setting"),
    [
        ("instance", "B"),
        ("instance", "D"),
        ("target", "C"),
        ("target", "D"),
    ],
)
def test_id_embeddings_reject_novel_entities(branch, validation_setting):
    config = make_valid_config()
    config["validation_setting"] = validation_setting
    config[f"{branch}_branch_architecture"] = "EMBEDDING"

    with pytest.raises(
        ConfigValidationError,
        match=rf"{branch}_branch_architecture='EMBEDDING'.*novel",
    ):
        DeepMTPConfig.from_mapping(config)


def test_id_embeddings_require_a_positive_vocabulary_size():
    config = make_valid_config(target_branch_architecture="EMBEDDING")
    config["target_branch_input_dim"] = 0

    with pytest.raises(
        ConfigValidationError,
        match="target_branch_input_dim",
    ):
        DeepMTPConfig.from_mapping(config)


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


def test_typed_config_rejects_rrmse_without_training_baseline_support():
    config = make_valid_config(problem_mode="regression", metrics=["MSE"])
    config["metrics"] = ["RRMSE"]

    with pytest.raises(
        ConfigValidationError,
        match="Unsupported regression metrics.*RRMSE",
    ):
        DeepMTPConfig.from_mapping(config)


def test_per_target_results_require_macro_averaging():
    config = make_valid_config(return_results_per_target=True)
    config["metrics_average"] = ["micro"]

    with pytest.raises(
        ConfigValidationError,
        match="return_results_per_target requires 'macro'",
    ):
        DeepMTPConfig.from_mapping(config)


def test_top_k_requires_grouped_averaging():
    config = make_valid_config(top_k=3)
    config["metrics_average"] = ["micro"]

    with pytest.raises(
        ConfigValidationError,
        match="top_k requires 'macro' or 'instance'",
    ):
        DeepMTPConfig.from_mapping(config)


@pytest.mark.parametrize(
    ("nodes_per_layer", "layer_count", "message"),
    [
        ([], None, "comb_mlp_nodes_per_layer"),
        ([8, 0], None, "comb_mlp_nodes_per_layer"),
        (0, 2, "comb_mlp_nodes_per_layer"),
        (8, None, "comb_mlp_layers"),
        (8, 0, "comb_mlp_layers"),
    ],
)
def test_combined_mlp_rejects_invalid_layer_settings(
    nodes_per_layer,
    layer_count,
    message,
):
    with pytest.raises(ConfigValidationError, match=message):
        make_valid_config(
            general_architecture_version="mlp",
            comb_mlp_nodes_per_layer=nodes_per_layer,
            comb_mlp_layers=layer_count,
        )


def test_combined_mlp_accepts_repeated_integer_width():
    config = make_valid_config(
        general_architecture_version="mlp",
        comb_mlp_nodes_per_layer=8,
        comb_mlp_layers=2,
    )

    typed_config = DeepMTPConfig.from_mapping(config)

    assert typed_config.comb_mlp_nodes_per_layer == 8
    assert typed_config.comb_mlp_layers == 2


@pytest.mark.parametrize("branch", ["instance", "target"])
@pytest.mark.parametrize(
    ("field_suffix", "invalid_value"),
    [
        ("conv_architecture", "alexnet"),
        ("conv_architecture_version", "resnet50"),
        ("conv_architecture_last_layer_trained", "classifier"),
        ("conv_architecture_dense_layers", 3),
        ("conv_architecture_dense_layers", True),
        ("conv_pretrained", "yes"),
    ],
)
def test_convolutional_branches_reject_unsupported_settings(
    branch,
    field_suffix,
    invalid_value,
):
    config = make_valid_config()
    config[f"{branch}_branch_architecture"] = "CONV"
    config[f"{branch}_branch_{field_suffix}"] = invalid_value

    with pytest.raises(
        ConfigValidationError,
        match=f"{branch}_branch_{field_suffix}",
    ):
        DeepMTPConfig.from_mapping(config)


def test_vgg_ignores_resnet_specific_legacy_settings():
    config = make_valid_config()
    config.update(
        {
            "instance_branch_architecture": "CONV",
            "instance_branch_conv_architecture": "VGG",
            "instance_branch_conv_architecture_version": "legacy-vgg-value",
            "instance_branch_conv_architecture_last_layer_trained": "legacy-value",
            "instance_branch_conv_architecture_dense_layers": 99,
        }
    )

    typed_config = DeepMTPConfig.from_mapping(config)

    assert typed_config.instance_branch_conv_architecture == "VGG"


def test_vgg_requires_boolean_pretrained_setting():
    config = make_valid_config()
    config.update(
        {
            "instance_branch_architecture": "CONV",
            "instance_branch_conv_architecture": "VGG",
            "instance_branch_conv_pretrained": 1,
        }
    )

    with pytest.raises(
        ConfigValidationError,
        match="instance_branch_conv_pretrained",
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
