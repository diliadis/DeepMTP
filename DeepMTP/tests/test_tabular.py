import numpy as np
import pandas as pd
import pytest
import torch

from DeepMTP.data import (
    TabularInput,
    TabularPreprocessingState,
    TabularPreprocessor,
    TabularSchema,
    data_process,
    process_instance_features,
)
from DeepMTP.models import TabularEncoder, encode_branch


def make_schema(**overrides):
    values = {
        "numeric_columns": ["age", "dose"],
        "categorical_columns": {
            "site": {
                "categories": ["A", "B"],
                "embedding_dim": 3,
            },
            "assay": {
                "categories": [10, 20, 30],
                "embedding_dim": 2,
            },
        },
        "numeric_normalization": "standard",
        "numeric_missing": "mean",
        "categorical_missing": "unknown",
        "categorical_unknown": "unknown",
        "feature_gating": True,
    }
    values.update(overrides)
    return TabularSchema.from_config(values, branch="instance")


def make_frame():
    return pd.DataFrame(
        {
            "id": [0, 1, 2],
            "age": [20.0, 22.0, np.nan],
            "dose": [1.0, 3.0, 5.0],
            "site": ["A", "B", None],
            "assay": [10, 20, 99],
        }
    )


def test_tabular_schema_has_stable_column_order_and_encoded_width():
    schema = make_schema()

    assert schema.numeric_columns == ("age", "dose")
    assert schema.categorical_names == ("site", "assay")
    assert schema.encoded_width == 7
    assert schema.categorical_columns[0].vocabulary_size == 3
    assert (
        TabularSchema.from_config(
            schema.to_dict(),
            branch="instance",
        )
        == schema
    )


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        (
            {"numeric_columns": [], "categorical_columns": {}},
            ValueError,
            "at least one",
        ),
        (
            {
                "numeric_columns": ["site"],
                "categorical_columns": {"site": ["A"]},
            },
            ValueError,
            "both numeric and categorical",
        ),
        (
            {"categorical_columns": {"site": []}},
            ValueError,
            "non-empty",
        ),
        (
            {"numeric_normalization": "robust"},
            ValueError,
            "numeric_normalization",
        ),
    ],
)
def test_tabular_schema_rejects_ambiguous_contracts(
    overrides,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        make_schema(**overrides)


def test_preprocessor_fits_training_statistics_and_reuses_them():
    schema = make_schema()
    train_frame = make_frame().iloc[:2]
    prediction_frame = make_frame().iloc[2:]

    fitted = TabularPreprocessor.fit(
        train_frame,
        schema,
        branch="instance",
    )
    restored = TabularPreprocessor.from_state(
        schema,
        fitted.state.to_dict(),
        branch="instance",
    )
    transformed = restored.transform(prediction_frame)
    features = transformed.iloc[0]["features"]

    assert fitted.state.numeric_fill_values == (21.0, 2.0)
    assert fitted.state.numeric_offsets == (21.0, 2.0)
    assert fitted.state.numeric_scales == (1.0, 1.0)
    assert features["numeric"].tolist() == pytest.approx([0.0, 3.0])
    assert features["categorical"].tolist() == [0, 0]


def test_preprocessor_state_rejects_a_different_schema():
    schema = make_schema()
    state = TabularPreprocessor.fit(
        make_frame().iloc[:2],
        schema,
        branch="instance",
    ).state.to_dict()
    state["numeric_columns"] = ["dose", "age"]

    with pytest.raises(ValueError, match="numeric columns do not match"):
        TabularPreprocessingState.from_config(
            state,
            schema=schema,
            branch="instance",
        )


def test_preprocessor_enforces_missing_and_unknown_policies():
    schema = make_schema(
        numeric_missing="error",
        categorical_missing="error",
        categorical_unknown="error",
    )

    with pytest.raises(ValueError, match="numeric_missing='error'"):
        TabularPreprocessor.fit(make_frame(), schema, branch="instance")

    clean_training = make_frame().iloc[:2]
    preprocessor = TabularPreprocessor.fit(
        clean_training,
        schema,
        branch="instance",
    )
    missing_category = make_frame().iloc[2:].copy()
    missing_category["age"] = 24.0
    with pytest.raises(ValueError, match="categorical_missing='error'"):
        preprocessor.transform(missing_category)

    unknown = make_frame().iloc[2:].copy()
    unknown["age"] = 24.0
    unknown["site"] = "A"
    with pytest.raises(ValueError, match="unknown value 99"):
        preprocessor.transform(unknown)


def test_feature_processing_preserves_explicit_tabular_columns():
    source = make_frame()

    processed = process_instance_features(source)

    assert processed["info"] == "tabular"
    assert processed["num_features"] == 4
    assert list(processed["data"].columns) == list(source.columns)
    assert processed["data"] is not source


def test_legacy_scaling_rejects_schema_shaped_tabular_features():
    scores = (np.arange(3)[:, np.newaxis] + np.arange(2)[np.newaxis, :]) % 2

    with pytest.raises(ValueError, match="numeric_normalization"):
        data_process(
            {
                "train": {
                    "y": scores,
                    "X_instance": make_frame(),
                    "X_target": None,
                }
            },
            validation_setting="B",
            scale_instance_features="Standard",
        )


def test_tabular_encoder_combines_numeric_and_categorical_features():
    schema = make_schema()
    encoder = TabularEncoder(
        {},
        schema,
        output_dim=4,
        nodes_per_layer=[5],
        num_layers=None,
        dropout_rate=0.0,
        batch_norm=False,
    )
    values = TabularInput(
        numeric=torch.tensor([[0.0, 1.0], [1.0, -1.0]]),
        categorical=torch.tensor([[1, 2], [2, 0]]),
    )

    encoded = encode_branch(encoder, values, branch="instance")

    assert encoder.input_kind == "tabular"
    assert encoder.output_dim == 4
    assert encoder.feature_gate is not None
    assert encoded.shape == (2, 4)


@pytest.mark.parametrize(
    ("schema_config", "values"),
    [
        (
            {
                "numeric_columns": ["age"],
                "categorical_columns": {},
            },
            TabularInput(
                numeric=torch.ones(2, 1),
                categorical=torch.empty(2, 0, dtype=torch.int64),
            ),
        ),
        (
            {
                "numeric_columns": [],
                "categorical_columns": {
                    "site": {
                        "categories": ["A", "B"],
                        "embedding_dim": 2,
                    }
                },
            },
            TabularInput(
                numeric=torch.empty(2, 0),
                categorical=torch.tensor([[1], [2]]),
            ),
        ),
    ],
)
def test_tabular_encoder_supports_single_modality_schemas(schema_config, values):
    schema = TabularSchema.from_config(schema_config, branch="instance")
    encoder = TabularEncoder(
        {},
        schema,
        output_dim=3,
        nodes_per_layer=[4],
        num_layers=None,
        dropout_rate=0.0,
        batch_norm=False,
    )

    assert encoder(values).shape == (2, 3)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            TabularInput(
                numeric=torch.ones(2, 1),
                categorical=torch.ones(2, 2, dtype=torch.int64),
            ),
            "numeric input width",
        ),
        (
            TabularInput(
                numeric=torch.ones(2, 2),
                categorical=torch.ones(2, 1, dtype=torch.int64),
            ),
            "categorical input width",
        ),
        (
            TabularInput(
                numeric=torch.ones(2, 2),
                categorical=torch.tensor([[1, 4], [2, 0]]),
            ),
            "outside its declared vocabulary",
        ),
    ],
)
def test_tabular_encoder_rejects_inputs_that_violate_the_schema(values, message):
    encoder = TabularEncoder(
        {},
        make_schema(),
        output_dim=4,
        nodes_per_layer=[5],
        num_layers=None,
        dropout_rate=0.0,
        batch_norm=False,
    )

    with pytest.raises(ValueError, match=message):
        encoder(values)
