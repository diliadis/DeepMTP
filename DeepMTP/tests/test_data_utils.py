import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from DeepMTP.dataset import (
    generate_MTP_dataset,
    process_dummy_DP,
    process_dummy_MLC,
    process_dummy_MTR,
)
from DeepMTP.utils.data_utils import (
    BaseDataset,
    MTPSampler,
    check_interaction_files_column_type_format,
    check_interaction_files_format,
    check_novel_instances,
    check_novel_targets,
    check_target_variable_type,
    check_variable_type,
    cross_input_consistency_check_instances,
    cross_input_consistency_check_targets,
    data_process,
    get_estimated_validation_setting,
    process_instance_features,
    process_interaction_data,
    process_target_features,
    split_data,
    validate_split_ratio,
)

data_format_type_combo_data = [
    {"pass_fail": "pass", "format": "numpy", "ids_type": "int"},
    {"pass_fail": "pass", "format": "dataframe", "ids_type": "int"},
    {
        "pass_fail": "fail",
        "format": "dataframe",
        "ids_type": "string",
        "data": pd.DataFrame(
            {
                "instance_id": ["a", "a", "b", "b"],
                "target_id": [0, 1, 0, 1],
                "value": [0, 1, 1, 0],
            }
        ),
    },
    {
        "pass_fail": "fail",
        "data": pd.DataFrame({"instance_id": [0, 0, 1, 1], "target_id": [0, 1, 0, 1]}),
    },  # less than three columns
    {
        "pass_fail": "fail",
        "data": pd.DataFrame(
            {"instance_id": [0, 0, 1, 1], "ta_id": [0, 1, 0, 1], "value": [0, 1, 1, 0]}
        ),
    },  # wrong column name
    {
        "pass_fail": "fail",
        "data": pd.DataFrame(
            {
                "instance_id": [0, 0, 1, 1],
                "target_id": [0, 1, 0, 1],
                "value": ["a", "b", "c", "d"],
            }
        ),
    },  # characters in the value column
    {
        "pass_fail": "fail",
        "data": pd.DataFrame(
            {
                "instance_id": [0.1, 0.2, 0.3, 0.4],
                "target_id": [0, 1, 0, 1],
                "value": [0, 1, 1, 0],
            }
        ),
    },  # wrong column name
    {
        "pass_fail": "fail",
        "data": pd.DataFrame(
            {
                "instance_id": [0, 0, 1, 1],
                "target_id": [0.1, 0.2, 0.3, 0.4],
                "value": [0, 1, 1, 0],
            }
        ),
    },  # wrong column name
    {"pass_fail": "fail", "data": "lalala"},  #
]


@pytest.mark.parametrize("data_format_type_combo_data", data_format_type_combo_data)
def test_process_interaction_data(data_format_type_combo_data):
    pass_fail = data_format_type_combo_data["pass_fail"]

    num_instances = 1000
    num_targets = 100
    num_instance_features = 2

    if pass_fail == "fail":
        with pytest.raises((AttributeError, TypeError, ValueError)):
            info = process_interaction_data(
                data_format_type_combo_data["data"], verbose=False
            )
    else:
        try:
            data_format = data_format_type_combo_data["format"]
            ids_type = data_format_type_combo_data["ids_type"]
            if ids_type == "int":
                data = process_dummy_MLC(
                    num_features=num_instance_features,
                    num_instances=num_instances,
                    num_targets=num_targets,
                    interaction_matrix_format=data_format,
                )
            else:
                data = {  # three instance features data sources and three interaction data matrices
                    "train": {
                        "y": pd.DataFrame(
                            {
                                "instance_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
                                "target_id": ["a", "b", "a", "b", "a", "b", "a", "b"],
                                "value": [0, 1, 0, 1, 0, 1, 0, 1],
                            }
                        ),
                        "X_instance": pd.DataFrame(
                            {
                                "id": ["a", "b", "c", "d"],
                                "features": list(np.random.rand(4, 10)),
                            }
                        ),
                        "X_target": pd.DataFrame(
                            {"id": ["a", "b"], "features": list(np.random.rand(2, 10))}
                        ),
                    },
                    "val": {
                        "y": pd.DataFrame(
                            {
                                "instance_id": ["e", "e", "f", "f"],
                                "target_id": ["a", "b", "a", "b"],
                                "value": [0, 1, 0, 1],
                            }
                        ),
                        "X_instance": pd.DataFrame(
                            {"id": ["e", "f"], "features": list(np.random.rand(2, 10))}
                        ),
                        "X_target": pd.DataFrame(
                            {"id": ["a", "b"], "features": list(np.random.rand(2, 10))}
                        ),
                    },
                    "test": {
                        "y": pd.DataFrame(
                            {
                                "instance_id": ["g", "g", "h", "h", "i", "i"],
                                "target_id": ["a", "b", "a", "b", "a", "b"],
                                "value": [0, 1, 0, 1, 0, 0],
                            }
                        ),
                        "X_instance": pd.DataFrame(
                            {
                                "id": ["g", "h", "i"],
                                "features": list(np.random.rand(3, 10)),
                            }
                        ),
                        "X_target": pd.DataFrame(
                            {"id": ["a", "b"], "features": list(np.random.rand(2, 10))}
                        ),
                    },
                }

            info = process_interaction_data(data["train"]["y"], verbose=False)
            assert (
                info["original_format"] == "numpy"
                if data_format == "numpy"
                else "triplets"
            )
            assert info["instance_id_type"] == ids_type
            assert info["target_id_type"] == ids_type

            if data_format == "dataframe":
                assert data["train"]["y"].equals(info["data"])
            else:
                triplets = [
                    (i, j, data["train"]["y"][i, j])
                    for i in range(data["train"]["y"].shape[0])
                    for j in range(data["train"]["y"].shape[1])
                ]
                temp_df = pd.DataFrame(
                    triplets, columns=["instance_id", "target_id", "value"]
                )
                # assert temp_df.equals(info["data"])
                assert_frame_equal(
                    temp_df.sort_values(by=temp_df.columns.tolist()).reset_index(
                        drop=True
                    ),
                    info["data"]
                    .sort_values(by=info["data"].columns.tolist())
                    .reset_index(drop=True),
                    check_dtype=False,
                )
        except Exception as exc:
            raise AssertionError("valid case unexpectedly failed") from exc


interaction_files_format_check_should_throw_error = [True, False]
test_check_interaction_files_format_data = [
    (
        "fail",
        {
            "train": {"y": {"original_format": "triplets"}},
            "val": {"y": {"original_format": "numpy"}},
            "test": {"y": {"original_format": "triplets"}},
        },
    ),
    (
        "pass",
        {
            "train": {"y": {"original_format": "triplets"}},
            "val": {"y": {"original_format": "triplets"}},
            "test": {"y": {"original_format": "triplets"}},
        },
    ),
    ("pass", {"train": {"y": None}, "val": {"y": None}, "test": {"y": None}}),
]


@pytest.mark.parametrize(
    "test_check_interaction_files_format_data", test_check_interaction_files_format_data
)
def test_check_interaction_files_format(test_check_interaction_files_format_data):
    pass_fail, data = test_check_interaction_files_format_data

    if pass_fail == "fail":
        with pytest.raises((AttributeError, TypeError, ValueError)):
            check_interaction_files_format(data)
    else:
        try:
            check_interaction_files_format(data)
        except Exception as exc:
            raise AssertionError("valid case unexpectedly failed") from exc


check_interaction_files_column_type_format_data = [
    (
        "pass",
        {
            "train": {"y": {"instance_id_type": "int", "target_id_type": "int"}},
            "val": {"y": {"instance_id_type": "int", "target_id_type": "int"}},
            "test": {"y": {"instance_id_type": "int", "target_id_type": "int"}},
        },
    ),
    (
        "pass",
        {
            "train": {"y": {"instance_id_type": "str", "target_id_type": "int"}},
            "val": {"y": {"instance_id_type": "str", "target_id_type": "int"}},
            "test": {"y": {"instance_id_type": "str", "target_id_type": "int"}},
        },
    ),
    (
        "fail",
        {
            "train": {"y": {"instance_id_type": "int", "target_id_type": "int"}},
            "val": {"y": {"instance_id_type": "str", "target_id_type": "int"}},
            "test": {"y": {"instance_id_type": "int", "target_id_type": "int"}},
        },
    ),
    (
        "fail",
        {
            "train": {"y": {"instance_id_type": "int", "target_id_type": "str"}},
            "val": {"y": {"instance_id_type": "int", "target_id_type": "int"}},
            "test": {"y": {"instance_id_type": "int", "target_id_type": "int"}},
        },
    ),
    ("pass", {"train": {"y": None}, "val": {"y": None}, "test": {"y": None}}),
]


@pytest.mark.parametrize(
    "check_interaction_files_column_type_format_data",
    check_interaction_files_column_type_format_data,
)
def test_check_interaction_files_column_type_format(
    check_interaction_files_column_type_format_data,
):
    pass_fail, data = check_interaction_files_column_type_format_data
    if pass_fail == "pass":
        try:
            check_interaction_files_column_type_format(data)
        except Exception as exc:
            raise AssertionError("valid case unexpectedly failed") from exc
    else:
        with pytest.raises((AttributeError, TypeError, ValueError)):
            check_interaction_files_column_type_format(data)


data_type = ["classification", "regression"]


@pytest.mark.parametrize("data_type", data_type)
def test_check_variable_type(data_type):
    num_instances = 1000
    num_targets = 100
    num_instance_features = 2

    if data_type == "classification":
        data = process_dummy_MLC(
            num_features=num_instance_features,
            num_instances=num_instances,
            num_targets=num_targets,
            interaction_matrix_format="dataframe",
        )
    elif data_type == "regression":
        data = process_dummy_MTR(
            num_features=num_instance_features,
            num_instances=num_instances,
            num_targets=num_targets,
            interaction_matrix_format="dataframe",
        )
    else:
        raise Exception(
            "Something went wrong. test_check_variable_type accepts only two values: "
            + str(["classification", "regression"])
        )

    if data_type == "classification":
        assert check_variable_type(data["train"]["y"]) == "binary"
    elif data_type == "regression":
        assert check_variable_type(data["train"]["y"]) == "real-valued"


classification_df = pd.DataFrame(
    {"instance_id": [0, 1, 2, 3], "target_id": [0, 1, 2, 3], "value": [0, 1, 0, 1]}
)
regression_df = pd.DataFrame(
    {
        "instance_id": [0, 1, 2, 3],
        "target_id": [0, 1, 2, 3],
        "value": [0.1, 0.2, 0.3, 0.4],
    }
)
check_target_variable_type_data = [
    (
        "pass",
        {
            "train": {"y": {"data": classification_df}},
            "val": {"y": {"data": classification_df}},
            "test": {"y": {"data": classification_df}},
            "mode": "binary",
        },
    ),
    (
        "pass",
        {
            "train": {"y": {"data": regression_df}},
            "val": {"y": {"data": regression_df}},
            "test": {"y": {"data": regression_df}},
            "mode": "real-valued",
        },
    ),
    (
        "fail",
        {
            "train": {"y": {"data": classification_df}},
            "val": {"y": {"data": regression_df}},
            "test": {"y": {"data": classification_df}},
        },
    ),
    ("fail", {"train": {"y": None}, "val": {"y": None}, "test": {"y": None}}),
]


@pytest.mark.parametrize(
    "check_target_variable_type_data", check_target_variable_type_data
)
def test_check_target_variable_type(check_target_variable_type_data):
    pass_fail, data = check_target_variable_type_data
    if pass_fail == "pass":
        assert check_target_variable_type(data) == data["mode"]
    else:
        with pytest.raises((AttributeError, TypeError, ValueError)):
            check_target_variable_type(data)


check_novel_instances_data = [
    (None, {"train": None, "test": None}),
    (
        True,
        {
            "train": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "triplets",
            },
            "test": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [4, 5, 6, 7],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 0, 0, 1],
                    }
                ),
                "original_format": "triplets",
            },
        },
    ),
    (
        False,
        {
            "train": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "triplets",
            },
            "test": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "triplets",
            },
        },
    ),
    (
        True,
        {
            "train": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "numpy",
            },
            "test": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [4, 5, 6, 7, 8],
                        "target_id": [0, 1, 2, 3, 4],
                        "value": [0, 0, 0, 1, 0],
                    }
                ),
                "original_format": "numpy",
            },
        },
    ),
    (
        False,
        {
            "train": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "numpy",
            },
            "test": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "numpy",
            },
        },
    ),
]


@pytest.mark.parametrize("check_novel_instances_data", check_novel_instances_data)
def test_check_novel_instances(check_novel_instances_data):
    true_false, data = check_novel_instances_data
    assert true_false == check_novel_instances(data["train"], data["test"])


check_novel_target_data = [
    (None, {"train": None, "test": None}),
    (
        True,
        {
            "train": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "triplets",
            },
            "test": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [4, 5, 6, 7],
                        "target_id": [4, 5, 6, 7],
                        "value": [0, 0, 0, 1],
                    }
                ),
                "original_format": "triplets",
            },
        },
    ),
    (
        False,
        {
            "train": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "triplets",
            },
            "test": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "triplets",
            },
        },
    ),
    (
        True,
        {
            "train": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "numpy",
            },
            "test": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [4, 5, 6, 7, 8],
                        "target_id": [4, 5, 6, 7, 8],
                        "value": [0, 0, 0, 1, 1],
                    }
                ),
                "original_format": "numpy",
            },
        },
    ),
    (
        False,
        {
            "train": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "numpy",
            },
            "test": {
                "data": pd.DataFrame(
                    {
                        "instance_id": [0, 1, 2, 3],
                        "target_id": [0, 1, 2, 3],
                        "value": [0, 1, 0, 1],
                    }
                ),
                "original_format": "numpy",
            },
        },
    ),
]


@pytest.mark.parametrize("check_novel_target_data", check_novel_target_data)
def test_check_novel_targets(check_novel_target_data):
    true_false, data = check_novel_target_data
    assert true_false == check_novel_targets(data["train"], data["test"])


get_estimated_validation_setting_data = [
    ((True, True), "D"),
    ((True, False), "B"),
    ((False, True), "C"),
    ((False, False), "A"),
    ((None, None), None),
]


@pytest.mark.parametrize(
    "get_estimated_validation_setting_data", get_estimated_validation_setting_data
)
def test_get_estimated_validation_setting(get_estimated_validation_setting_data):
    (
        novel_instance_targets_tuple,
        true_validation_setting,
    ) = get_estimated_validation_setting_data
    assert (
        get_estimated_validation_setting(
            novel_instance_targets_tuple[0],
            novel_instance_targets_tuple[1],
            verbose=False,
        )
        == true_validation_setting
    )


test_process_instance_features_data = [
    {"format": "numpy", "ids_type": "int", "features_type": "features"},
    {"format": "dataframe", "ids_type": "int", "features_type": "features"},
    # {'format': 'dataframe', 'ids_type': 'string', 'features_type': 'features'},
    {"format": "None", "ids_type": "None", "features_type": "features"},
    # {'format': 'dataframe', 'ids_type': 'string', 'features_type': 'dir'},
]


@pytest.mark.parametrize(
    "test_process_instance_features_data", test_process_instance_features_data
)
def test_process_instance_features(test_process_instance_features_data):
    num_instances = 1000
    num_targets = 100
    num_instance_features = 10

    data_format = test_process_instance_features_data["format"]
    ids_type = test_process_instance_features_data["ids_type"]

    if ids_type == "int":
        data = process_dummy_MLC(
            num_features=num_instance_features,
            num_instances=num_instances,
            num_targets=num_targets,
            interaction_matrix_format="numpy",
            features_format=data_format,
        )
        if data_format == "numpy":
            original_instance_features = pd.DataFrame(
                np.arange(len(data["train"]["X_instance"])), columns=["id"]
            )
            original_instance_features["features"] = [
                r for r in data["train"]["X_instance"]
            ]
    """
    else:
        if features_type == 'features':
            data = { # three instance features data sources and three interaction data matrices
            'train': {
                'y': pd.DataFrame({'instance_id': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], 'target_id': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1, 0, 1, 0, 1]}),
                'X_instance': pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'features': list(np.random.rand(4, 10))}),
                'X_target': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            },
            'val': {
                'y': pd.DataFrame({'instance_id': ['e', 'e', 'f', 'f'], 'target_id': ['a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1]}),
                'X_instance': pd.DataFrame({'id': ['e', 'f'], 'features': list(np.random.rand(2, 10))}),
                'X_target': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            },
            'test': {
                'y': pd.DataFrame({'instance_id': ['g', 'g', 'h', 'h', 'i', 'i'], 'target_id': ['a', 'b', 'a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1, 0, 0]}),
                'X_instance': pd.DataFrame({'id': ['g', 'h', 'i'], 'features': list(np.random.rand(3, 10))}),
                'X_target': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            }}
        else:
            data = { # three instance features data sources and three interaction data matrices
            'train': {
                'y': pd.DataFrame({'instance_id': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], 'target_id': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1, 0, 1, 0, 1]}),
                'X_instance': pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'dir': ['testing', 'is', 'really', 'hard']}),
                'X_target': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            },
            'val': {
                'y': pd.DataFrame({'instance_id': ['e', 'e', 'f', 'f'], 'target_id': ['a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1]}),
                'X_instance': pd.DataFrame({'id': ['e', 'f'], 'features': list(np.random.rand(2, 10))}),
                'X_target': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            },
            'test': {
                'y': pd.DataFrame({'instance_id': ['g', 'g', 'h', 'h', 'i', 'i'], 'target_id': ['a', 'b', 'a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1, 0, 0]}),
                'X_instance': pd.DataFrame({'id': ['g', 'h', 'i'], 'features': list(np.random.rand(3, 10))}),
                'X_target': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            }}
    """
    if ids_type in ["int", "string"]:
        instance_features = process_instance_features(
            data["train"]["X_instance"], verbose=False
        )
        if "dir" not in instance_features["data"].columns:
            assert instance_features["num_features"] == num_instance_features
            assert instance_features["info"] == data_format

        if data_format == "numpy":
            assert original_instance_features.equals(instance_features["data"])
        else:
            assert data["train"]["X_instance"].equals(instance_features["data"])
    else:
        assert process_instance_features(None, verbose=False) is None


test_process_target_features_data = [
    {"format": "numpy", "ids_type": "int", "features_type": "features"},
    {"format": "dataframe", "ids_type": "int", "features_type": "features"},
    # {'format': 'dataframe', 'ids_type': 'string', 'features_type': 'features'},
    {"format": "None", "ids_type": "None", "features_type": "features"},
    # {'format': 'dataframe', 'ids_type': 'string', 'features_type': 'dir'},
]


@pytest.mark.parametrize(
    "test_process_target_features_data", test_process_target_features_data
)
def test_process_target_features(test_process_target_features_data):
    num_instances = 1000
    num_targets = 100
    num_instance_features = 20
    num_target_features = 10

    data_format = test_process_target_features_data["format"]
    ids_type = test_process_target_features_data["ids_type"]

    if ids_type == "int":
        data = process_dummy_DP(
            num_instance_features=num_instance_features,
            num_target_features=num_target_features,
            num_instances=num_instances,
            num_targets=num_targets,
            interaction_matrix_format="numpy",
            instance_features_format="numpy",
            target_features_format=data_format,
        )
        if data_format == "numpy":
            original_target_features = pd.DataFrame(
                np.arange(len(data["train"]["X_target"])), columns=["id"]
            )
            original_target_features["features"] = [
                r for r in data["train"]["X_target"]
            ]
    """
    else:
        if features_type == 'features':
            data = { # three instance features data sources and three interaction data matrices
            'train': {
                'y': pd.DataFrame({'target_id': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], 'instance_id': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1, 0, 1, 0, 1]}),
                'X_target': pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'features': list(np.random.rand(4, 10))}),
                'X_instance': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            },
            'val': {
                'y': pd.DataFrame({'target_id': ['e', 'e', 'f', 'f'], 'instance_id': ['a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1]}),
                'X_target': pd.DataFrame({'id': ['e', 'f'], 'features': list(np.random.rand(2, 10))}),
                'X_instance': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            },
            'test': {
                'y': pd.DataFrame({'target_id': ['g', 'g', 'h', 'h', 'i', 'i'], 'instance_id': ['a', 'b', 'a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1, 0, 0]}),
                'X_target': pd.DataFrame({'id': ['g', 'h', 'i'], 'features': list(np.random.rand(3, 10))}),
                'X_instance': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            }}
        else:
            data = { # three instance features data sources and three interaction data matrices
            'train': {
                'y': pd.DataFrame({'target_id': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], 'instance_id': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1, 0, 1, 0, 1]}),
                'X_target': pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'dir': ['testing', 'is', 'really', 'hard']}),
                'X_instance': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            },
            'val': {
                'y': pd.DataFrame({'target_id': ['e', 'e', 'f', 'f'], 'instance_id': ['a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1]}),
                'X_target': pd.DataFrame({'id': ['e', 'f'], 'features': list(np.random.rand(2, 10))}),
                'X_instance': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            },
            'test': {
                'y': pd.DataFrame({'target_id': ['g', 'g', 'h', 'h', 'i', 'i'], 'instance_id': ['a', 'b', 'a', 'b', 'a', 'b'], 'value': [0, 1, 0, 1, 0, 0]}),
                'X_target': pd.DataFrame({'id': ['g', 'h', 'i'], 'features': list(np.random.rand(3, 10))}),
                'X_instance': pd.DataFrame({'id': ['a', 'b'], 'features': list(np.random.rand(2, 10))}),
            }}
    """
    if ids_type in ["int", "string"]:
        target_features = process_target_features(
            data["train"]["X_target"], verbose=False
        )
        if "dir" not in target_features["data"].columns:
            assert target_features["num_features"] == num_target_features
            assert target_features["info"] == data_format

        if data_format == "numpy":
            assert original_target_features.equals(target_features["data"])
        else:
            assert data["train"]["X_target"].equals(target_features["data"])
    else:
        assert process_target_features(None, verbose=False) is None


test_cross_input_consistency_check_instances_data = [
    (
        "pass",
        {
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [4, 5], "features": list(np.random.rand(2, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "fail",
        {  # the instance features in the validation set are less than what is required in the interaction matrix
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [4], "features": list(np.random.rand(1, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "fail",
        {  # 3 interaction files but 2 instance feature files (test instance features are missing)
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "fail",
        {  # more than one instance features files while on setting A)
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "A",
        },
    ),
    (
        "pass",
        {  # more than one instance features files while on setting A)
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 4, 5, 5, 5, 6, 6],
                            "target_id": [0, 2, 1, 3, 2, 4, 1, 0, 2, 4, 0, 2],
                            "value": [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5, 6],
                            "features": list(np.random.rand(7, 10)),
                        }
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6],
                            "target_id": [1, 3, 0, 2, 0, 1, 1, 2, 3, 1, 3, 1, 4],
                            "value": [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": None,
                "X_target": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 2, 3, 3, 3, 3, 4, 4, 6],
                            "target_id": [4, 4, 3, 0, 2, 3, 4, 0, 4, 3],
                            "value": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": None,
                "X_target": None,
            },
            "validation_setting": "A",
        },
    ),
    # triplets version
    (
        "fail",
        {  # the instance features in the validation set are less than what is required in the interaction matrix
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": None,
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [4, 5], "features": list(np.random.rand(2, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "pass",
        {  # three instance features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [4, 5], "features": list(np.random.rand(2, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "fail",
        {  # two instance features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5],
                            "features": list(np.random.rand(6, 10)),
                        }
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "pass",
        {  # one instance features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                            "features": list(np.random.rand(9, 10)),
                        }
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "pass",
        {  # one instance features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "fail",
        {  # more than one instance feature data sources while on setting C
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5],
                            "features": list(np.random.rand(6, 10)),
                        }
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "C",
        },
    ),
    (
        "pass",
        {
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                            "features": list(np.random.rand(9, 10)),
                        }
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "C",
        },
    ),
]


@pytest.mark.parametrize(
    "test_cross_input_consistency_check_instances_data",
    test_cross_input_consistency_check_instances_data,
)
def test_cross_input_consistency_check_instances(
    test_cross_input_consistency_check_instances_data,
):
    pass_fail, data = test_cross_input_consistency_check_instances_data
    if pass_fail == "pass":
        try:
            cross_input_consistency_check_instances(
                data, data["validation_setting"], verbose=False
            )
        except Exception as exc:
            raise AssertionError("valid case unexpectedly failed") from exc
    else:
        with pytest.raises((AttributeError, TypeError, ValueError)):
            cross_input_consistency_check_instances(
                data, data["validation_setting"], verbose=False
            )


test_cross_input_consistency_check_targets_data = [
    (
        "pass",
        {
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [4, 5], "features": list(np.random.rand(2, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "C",
        },
    ),
    (
        "pass",
        {
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "target_id": [4, 4, 5, 5],
                            "instance_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [4, 5], "features": list(np.random.rand(2, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "C",
        },
    ),
    (
        "fail",
        {  # 3 interaction files but 2 instance feature files (test instance features are missing)
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": None,
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "C",
        },
    ),
    (
        "fail",
        {  # 3 interaction files but 2 instance feature files (test instance features are missing)
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": None,
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": None,
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "C",
        },
    ),
    (
        "fail",
        {  # more than one instance features files while on setting A)
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": None,
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "A",
        },
    ),
    (
        "pass",
        {  # more than one instance features files while on setting A)
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "target_id": [0, 0, 1, 1, 2, 2, 4, 5, 5, 5, 6, 6],
                            "instance_id": [0, 2, 1, 3, 2, 4, 1, 0, 2, 4, 0, 2],
                            "value": [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5, 6],
                            "features": list(np.random.rand(7, 10)),
                        }
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6],
                            "instance_id": [1, 3, 0, 2, 0, 1, 1, 2, 3, 1, 3, 1, 4],
                            "value": [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": None,
                "X_instance": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "target_id": [0, 1, 2, 3, 3, 3, 3, 4, 4, 6],
                            "instance_id": [4, 4, 3, 0, 2, 3, 4, 0, 4, 3],
                            "value": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        }
                    ),
                    "original_format": "numpy",
                },
                "X_target": None,
                "X_instance": None,
            },
            "validation_setting": "A",
        },
    ),
    # triplets version
    (
        "fail",
        {  # the target features in the validation set are less than what is required in the interaction matrix
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [4, 5], "features": list(np.random.rand(2, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "C",
        },
    ),
    (
        "pass",
        {  # three instance features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1, 2, 3], "features": list(np.random.rand(4, 10))}
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1],
                            "target_id": [4, 4, 5, 5],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [4, 5], "features": list(np.random.rand(2, 10))}
                    )
                },
                "X_instance": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_instance": None,
            },
            "validation_setting": "C",
        },
    ),
    (
        "fail",
        {  # two target features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5],
                            "features": list(np.random.rand(6, 10)),
                        }
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1],
                            "target_id": [4, 4, 5, 5],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": None,
                "X_instance": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_instance": None,
            },
            "validation_setting": "C",
        },
    ),
    (
        "fail",
        {  # two target features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5],
                            "features": list(np.random.rand(6, 10)),
                        }
                    )
                },
                "X_instance": None,
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1],
                            "target_id": [4, 4, 5, 5],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": None,
                "X_instance": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [6, 7, 8], "features": list(np.random.rand(3, 10))}
                    )
                },
                "X_instance": None,
            },
            "validation_setting": "C",
        },
    ),
    (
        "pass",
        {  # one target features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": {
                    "data": pd.DataFrame(
                        {
                            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                            "features": list(np.random.rand(9, 10)),
                        }
                    )
                },
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1],
                            "target_id": [4, 4, 5, 5],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": None,
                "X_instance": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": None,
                "X_instance": None,
            },
            "validation_setting": "C",
        },
    ),
    (
        "pass",
        {  # one target features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "target_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": None,
                "X_instance": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1],
                            "target_id": [4, 4, 5, 5],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": None,
                "X_instance": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 1, 0, 1, 0, 1],
                            "target_id": [6, 6, 7, 7, 8, 8],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_target": None,
                "X_instance": None,
            },
            "validation_setting": "C",
        },
    ),
    (
        "fail",
        {  # one instance features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "validation_setting": "B",
        },
    ),
    (
        "pass",
        {  # one instance features data sources and three interaction data matrices
            "train": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [0, 0, 1, 1, 2, 2, 3, 3],
                            "target_id": [0, 1, 0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": {
                    "data": pd.DataFrame(
                        {"id": [0, 1], "features": list(np.random.rand(2, 10))}
                    )
                },
            },
            "val": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [4, 4, 5, 5],
                            "target_id": [0, 1, 0, 1],
                            "value": [0, 1, 0, 1],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": None,
            },
            "test": {
                "y": {
                    "data": pd.DataFrame(
                        {
                            "instance_id": [6, 6, 7, 7, 8, 8],
                            "target_id": [0, 1, 0, 1, 0, 1],
                            "value": [0, 1, 0, 1, 0, 0],
                        }
                    ),
                    "original_format": "triplets",
                },
                "X_instance": None,
                "X_target": None,
            },
            "validation_setting": "B",
        },
    ),
]


@pytest.mark.parametrize(
    "test_cross_input_consistency_check_targets_data",
    test_cross_input_consistency_check_targets_data,
)
def test_cross_input_consistency_check_targets(
    test_cross_input_consistency_check_targets_data,
):
    pass_fail, data = test_cross_input_consistency_check_targets_data
    if pass_fail == "pass":
        try:
            cross_input_consistency_check_targets(
                data, data["validation_setting"], verbose=False
            )
        except Exception as exc:
            raise AssertionError("valid case unexpectedly failed") from exc
    else:
        with pytest.raises((AttributeError, TypeError, ValueError)):
            cross_input_consistency_check_targets(
                data, data["validation_setting"], verbose=False
            )


test_split_data_data = [
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": None,
        "split_targets": None,
        "validation_setting": "B",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": None,
        "split_instances": None,
        "split_targets": None,
        "validation_setting": "B",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": {"train": 0.7, "val": 0.1, "test": 0.3},
        "split_targets": None,
        "validation_setting": "B",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": None,
        "split_instances": {"train": 0.7, "val": 0.1, "test": 0.3},
        "split_targets": None,
        "validation_setting": "B",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": {"train": 0.7, "test": 0.3},
        "split_targets": None,
        "validation_setting": "B",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": None,
        "split_instances": {"train": 0.7, "test": 0.3},
        "split_targets": None,
        "validation_setting": "B",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": None,
        "split_targets": None,
        "validation_setting": "C",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": None,
        "num_target_features": 5,
        "split_instances": None,
        "split_targets": None,
        "validation_setting": "C",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": None,
        "split_targets": {"train": 0.7, "val": 0.1, "test": 0.3},
        "validation_setting": "C",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": None,
        "num_target_features": 5,
        "split_instances": None,
        "split_targets": {"train": 0.7, "val": 0.1, "test": 0.3},
        "validation_setting": "C",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": None,
        "split_targets": {"train": 0.7, "test": 0.3},
        "validation_setting": "C",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": None,
        "num_target_features": 5,
        "split_instances": None,
        "split_targets": {"train": 0.7, "test": 0.3},
        "validation_setting": "C",
    },
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": None,
        "split_targets": None,
        "validation_setting": "D",
    },
    # {'num_instances': 100, 'num_targets': 15, 'num_instance_features': 10, 'num_target_features': 5, 'split_instances': {'train':0.7, 'val':0.1, 'test':0.3}, 'split_targets': None, 'validation_setting': 'D'},
    # {'num_instances': 100, 'num_targets': 15, 'num_instance_features': 10, 'num_target_features': 5, 'split_instances': {'train':0.7, 'test':0.3}, 'split_targets': None, 'validation_setting': 'D'},
    # {'num_instances': 100, 'num_targets': 15, 'num_instance_features': 10, 'num_target_features': 5, 'split_instances': None, 'split_targets': {'train':0.7, 'val':0.1, 'test':0.3}, 'validation_setting': 'D'},
    # {'num_instances': 100, 'num_targets': 15, 'num_instance_features': 10, 'num_target_features': 5, 'split_instances': None, 'split_targets': {'train':0.7, 'test':0.3}, 'validation_setting': 'D'},
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": {"train": 0.7, "val": 0.1, "test": 0.3},
        "split_targets": {"train": 0.7, "val": 0.1, "test": 0.3},
        "validation_setting": "D",
    },
    # {'num_instances': 100, 'num_targets': 15, 'num_instance_features': 10, 'num_target_features': 5, 'split_instances': {'train':0.7, 'test':0.3}, 'split_targets': {'train':0.7, 'val':0.1, 'test':0.3}, 'validation_setting': 'D'},
    # {'num_instances': 100, 'num_targets': 15, 'num_instance_features': 10, 'num_target_features': 5, 'split_instances': {'train':0.7, 'val':0.1, 'test':0.3}, 'split_targets': {'train':0.7, 'test':0.3}, 'validation_setting': 'D'},
    {
        "num_instances": 100,
        "num_targets": 15,
        "num_instance_features": 10,
        "num_target_features": 5,
        "split_instances": {"train": 0.7, "test": 0.3},
        "split_targets": {"train": 0.7, "test": 0.3},
        "validation_setting": "D",
    },
]


@pytest.mark.parametrize("test_split_data_data", test_split_data_data)
def test_split_data(test_split_data_data):
    original_data = generate_MTP_dataset(
        test_split_data_data["num_instances"],
        test_split_data_data["num_targets"],
        num_instance_features=test_split_data_data["num_instance_features"],
        num_target_features=test_split_data_data["num_target_features"],
        split_instances={"train": 0.7, "val": 0.1, "test": 0.3}
        if test_split_data_data["validation_setting"] in ["B", "D"]
        else None,
        split_targets={"train": 0.7, "val": 0.1, "test": 0.3}
        if test_split_data_data["validation_setting"] in ["C", "D"]
        else None,
        return_static_features_data=True,
    )

    processed_data = generate_MTP_dataset(
        test_split_data_data["num_instances"],
        test_split_data_data["num_targets"],
        num_instance_features=test_split_data_data["num_instance_features"],
        num_target_features=test_split_data_data["num_target_features"],
        split_instances=test_split_data_data["split_instances"],
        split_targets=test_split_data_data["split_targets"],
        return_static_features_data=True,
    )

    split_data(
        processed_data,
        test_split_data_data["validation_setting"],
        split_method="random",
        ratio={"train": 0.6, "val": 0.1, "test": 0.3},
        shuffle=False,
        seed=42,
        verbose=False,
        print_mode="basic",
    )

    assert len(original_data) == len(processed_data)
    for mode in original_data:
        print("mode: " + str(mode))
        assert original_data[mode]["y"]["data"].equals(
            processed_data[mode]["y"]["data"]
        )
        if original_data[mode]["X_instance"]:
            assert original_data[mode]["X_instance"]["data"].equals(
                processed_data[mode]["X_instance"]["data"]
            )
        if original_data[mode]["X_target"]:
            assert original_data[mode]["X_target"]["data"].equals(
                processed_data[mode]["X_target"]["data"]
            )


@pytest.mark.parametrize("validation_setting", ["B", "C", "D"])
def test_split_data_recovers_missing_test_side_features(validation_setting):
    split_instances = (
        {"train": 0.7, "test": 0.3} if validation_setting in {"B", "D"} else None
    )
    split_targets = (
        {"train": 0.7, "test": 0.3} if validation_setting in {"C", "D"} else None
    )
    data = generate_MTP_dataset(
        20,
        10,
        num_instance_features=3,
        num_target_features=2,
        split_instances=split_instances,
        split_targets=split_targets,
        return_static_features_data=True,
    )

    feature_axes = []
    if split_instances is not None:
        feature_axes.append(("X_instance", "instance_id"))
    if split_targets is not None:
        feature_axes.append(("X_target", "target_id"))

    for feature_key, _ in feature_axes:
        data["train"][feature_key]["data"] = (
            pd.concat(
                [
                    data["train"][feature_key]["data"],
                    data["test"][feature_key]["data"],
                ],
                ignore_index=True,
            )
            .drop_duplicates(subset="id")
            .sort_values("id")
        )
        data["test"][feature_key] = None
        data["val"][feature_key] = None

    split_data(
        data,
        validation_setting,
        split_method="random",
        ratio={"train": 0.6, "val": 0.1, "test": 0.3},
        shuffle=False,
        seed=42,
        verbose=False,
    )

    for feature_key, interaction_id in feature_axes:
        for split_name in ("train", "val", "test"):
            feature_ids = set(data[split_name][feature_key]["data"]["id"])
            interaction_ids = set(
                data[split_name]["y"]["data"][interaction_id].unique()
            )
            assert feature_ids == interaction_ids


test_data_process_data = [
    {
        "pass_fail": "pass",
        "MTP_setting": "MLC",
        "interaction_matrix_format": "numpy",
        "scale_instance_features": None,
        "scale_target_features": None,
    },
    {
        "pass_fail": "pass",
        "MTP_setting": "MLC",
        "interaction_matrix_format": "numpy",
        "scale_instance_features": "MinMax",
        "scale_target_features": None,
    },
    {
        "pass_fail": "pass",
        "MTP_setting": "MLC",
        "interaction_matrix_format": "numpy",
        "scale_instance_features": "MinMax",
        "scale_target_features": "MinMax",
    },
    {
        "pass_fail": "pass",
        "MTP_setting": "MTR",
        "interaction_matrix_format": "dataframe",
        "scale_instance_features": None,
        "scale_target_features": None,
    },
    {
        "pass_fail": "pass",
        "MTP_setting": "DP",
        "interaction_matrix_format": "numpy",
        "scale_instance_features": None,
        "scale_target_features": None,
    },
    {
        "pass_fail": "pass",
        "MTP_setting": "DP",
        "interaction_matrix_format": "numpy",
        "scale_instance_features": "Standard",
        "scale_target_features": "MinMax",
    },
    {"pass_fail": "fail", "data": {"test": {}, "val": {}}},
    {"pass_fail": "fail", "data": {"train": {}, "test": {}, "val": {}}},
    {"pass_fail": "fail", "data": {"train": {}}},
    {"pass_fail": "fail", "data": {"train": {"y": {}}, "val": {"y": {}}}},
    # {'pass_fail': 'pass', 'MTP_setting': 'DP'},
    {"pass_fail": "fail", "data": {"train": {"y": {}}}},
]


@pytest.mark.parametrize("test_data_process_data", test_data_process_data)
def test_data_process(test_data_process_data):
    pass_fail = test_data_process_data["pass_fail"]

    num_instances = 100
    num_targets = 20
    num_instance_features = 15
    num_target_features = 10

    if pass_fail == "pass":
        try:
            MTP_setting = test_data_process_data["MTP_setting"]
            interaction_matrix_format = test_data_process_data[
                "interaction_matrix_format"
            ]
            scale_instance_features = test_data_process_data["scale_instance_features"]
            scale_target_features = test_data_process_data["scale_target_features"]
            validation_setting = None

            if MTP_setting == "MLC":
                data = process_dummy_MLC(
                    num_features=num_instance_features,
                    num_instances=num_instances,
                    num_targets=num_targets,
                    interaction_matrix_format=interaction_matrix_format,
                    features_format="numpy",
                )
                validation_setting = "B"
            elif MTP_setting == "MTR":
                data = process_dummy_MTR(
                    num_features=num_instance_features,
                    num_instances=num_instances,
                    num_targets=num_targets,
                    interaction_matrix_format=interaction_matrix_format,
                    features_format="dataframe",
                    variant="divided",
                    split_ratio={"train": 0.7, "val": 0.1, "test": 0.2},
                    random_state=42,
                )
            elif MTP_setting == "DP":
                data = process_dummy_DP(
                    num_instance_features=num_instance_features,
                    num_target_features=num_target_features,
                    num_instances=num_instances,
                    num_targets=num_targets,
                    interaction_matrix_format=interaction_matrix_format,
                    instance_features_format="dataframe",
                    target_features_format="dataframe",
                )
                validation_setting = "D"

            train, val, test, info = data_process(
                data,
                validation_setting=validation_setting,
                split_method="random",
                ratio={"train": 0.7, "test": 0.2, "val": 0.1},
                shuffle=True,
                seed=42,
                verbose=False,
                print_mode="basic",
                scale_instance_features=scale_instance_features,
                scale_target_features=scale_target_features,
            )

            if info["detected_validation_setting"] == "B":
                assert len(train["y"]["data"]) == (
                    len(set(train["y"]["data"]["instance_id"]))
                    * len(set(train["y"]["data"]["target_id"]))
                )
                assert len(val["y"]["data"]) == (
                    len(set(val["y"]["data"]["instance_id"]))
                    * len(set(val["y"]["data"]["target_id"]))
                )
                assert len(test["y"]["data"]) == (
                    len(set(test["y"]["data"]["instance_id"]))
                    * len(set(test["y"]["data"]["target_id"]))
                )

                if interaction_matrix_format == "dataframe":
                    # no overlaping instance ids between the train, val and test sets
                    assert (
                        set(train["y"]["data"]["instance_id"]).intersection(
                            set(val["y"]["data"]["instance_id"])
                        )
                        == set()
                    )
                    assert (
                        set(train["y"]["data"]["instance_id"]).intersection(
                            set(test["y"]["data"]["instance_id"])
                        )
                        == set()
                    )
                    assert (
                        set(val["y"]["data"]["instance_id"]).intersection(
                            set(test["y"]["data"]["instance_id"])
                        )
                        == set()
                    )

                # in setting B you don't split on targets, so train, val and test should have the same target ids
                assert set(train["y"]["data"]["target_id"]) == set(
                    val["y"]["data"]["target_id"]
                )
                assert set(train["y"]["data"]["target_id"]) == set(
                    test["y"]["data"]["target_id"]
                )

                # train,val and test have the same instance ids between their interaction matrices and instance features
                assert set(train["X_instance"]["data"]["id"]) == set(
                    train["y"]["data"]["instance_id"]
                )
                assert set(val["X_instance"]["data"]["id"]) == set(
                    val["y"]["data"]["instance_id"]
                )
                assert set(test["X_instance"]["data"]["id"]) == set(
                    test["y"]["data"]["instance_id"]
                )

                if train["X_target"] is not None:
                    assert set(train["X_target"]["data"]["id"]) == set(
                        train["y"]["data"]["target_id"]
                    )
                    assert set(val["X_target"]["data"]["id"]) == set(
                        val["y"]["data"]["target_id"]
                    )
                    assert set(test["X_target"]["data"]["id"]) == set(
                        test["y"]["data"]["target_id"]
                    )

            elif info["detected_validation_setting"] == "C":
                assert len(train["y"]["data"]) == (
                    len(set(train["y"]["data"]["instance_id"]))
                    * len(set(train["y"]["data"]["target_id"]))
                )
                assert len(val["y"]["data"]) == (
                    len(set(val["y"]["data"]["instance_id"]))
                    * len(set(val["y"]["data"]["target_id"]))
                )
                assert len(test["y"]["data"]) == (
                    len(set(test["y"]["data"]["instance_id"]))
                    * len(set(test["y"]["data"]["target_id"]))
                )

                if interaction_matrix_format == "dataframe":
                    # no overlaping instance ids between the train, val and test sets
                    assert (
                        set(train["y"]["data"]["target_id"]).intersection(
                            set(val["y"]["data"]["target_id"])
                        )
                        == set()
                    )
                    assert (
                        set(train["y"]["data"]["target_id"]).intersection(
                            set(test["y"]["data"]["target_id"])
                        )
                        == set()
                    )
                    assert (
                        set(val["y"]["data"]["target_id"]).intersection(
                            set(test["y"]["data"]["target_id"])
                        )
                        == set()
                    )

                # in setting B you don't split on targets, so train, val and test should have the same target ids
                assert set(train["y"]["data"]["instance_id"]) == set(
                    val["y"]["data"]["instance_id"]
                )
                assert set(train["y"]["data"]["instance_id"]) == set(
                    test["y"]["data"]["instance_id"]
                )

                # train,val and test have the same instance ids between their interaction matrices and instance features
                assert set(train["X_target"]["data"]["id"]) == set(
                    train["y"]["data"]["target_id"]
                )
                assert set(val["X_target"]["data"]["id"]) == set(
                    val["y"]["data"]["target_id"]
                )
                assert set(test["X_target"]["data"]["id"]) == set(
                    test["y"]["data"]["target_id"]
                )

                if train["X_target"] is not None:
                    assert set(train["X_instance"]["data"]["id"]) == set(
                        train["y"]["data"]["instance_id"]
                    )
                    assert set(val["X_instance"]["data"]["id"]) == set(
                        val["y"]["data"]["instance_id"]
                    )
                    assert set(test["X_instance"]["data"]["id"]) == set(
                        test["y"]["data"]["instance_id"]
                    )

            elif info["detected_validation_setting"] == "D":
                assert len(train["y"]["data"]) == (
                    len(set(train["y"]["data"]["instance_id"]))
                    * len(set(train["y"]["data"]["target_id"]))
                )
                assert len(val["y"]["data"]) == (
                    len(set(val["y"]["data"]["instance_id"]))
                    * len(set(val["y"]["data"]["target_id"]))
                )
                assert len(test["y"]["data"]) == (
                    len(set(test["y"]["data"]["instance_id"]))
                    * len(set(test["y"]["data"]["target_id"]))
                )
                if interaction_matrix_format == "dataframe":
                    # no overlaping instance ids between the train, val and test sets
                    assert (
                        set(train["y"]["data"]["instance_id"]).intersection(
                            set(val["y"]["data"]["instance_id"])
                        )
                        == set()
                    )
                    assert (
                        set(train["y"]["data"]["instance_id"]).intersection(
                            set(test["y"]["data"]["instance_id"])
                        )
                        == set()
                    )
                    assert (
                        set(val["y"]["data"]["instance_id"]).intersection(
                            set(test["y"]["data"]["instance_id"])
                        )
                        == set()
                    )

                # train,val and test have the same instance ids between their interaction matrices and instance features
                assert set(train["X_instance"]["data"]["id"]) == set(
                    train["y"]["data"]["instance_id"]
                )
                assert set(val["X_instance"]["data"]["id"]) == set(
                    val["y"]["data"]["instance_id"]
                )
                assert set(test["X_instance"]["data"]["id"]) == set(
                    test["y"]["data"]["instance_id"]
                )
                # train,val and test have the same instance ids between their interaction matrices and instance features
                assert set(train["X_target"]["data"]["id"]) == set(
                    train["y"]["data"]["target_id"]
                )
                assert set(val["X_target"]["data"]["id"]) == set(
                    val["y"]["data"]["target_id"]
                )
                assert set(test["X_target"]["data"]["id"]) == set(
                    test["y"]["data"]["target_id"]
                )

        except Exception as exc:
            raise AssertionError("valid case unexpectedly failed") from exc
    else:
        with pytest.raises((AttributeError, TypeError, ValueError)):
            train, val, test, info = data_process(
                test_data_process_data["data"],
                validation_setting=None,
                split_method="random",
                ratio={"train": 0.7, "test": 0.2, "val": 0.1},
                shuffle=True,
                seed=42,
                verbose=False,
                print_mode="basic",
                scale_instance_features=None,
                scale_target_features=None,
            )


def test_data_process_default_ratio_is_reusable():
    data = process_dummy_MLC(
        num_features=5,
        num_instances=100,
        num_targets=4,
        interaction_matrix_format="numpy",
        features_format="numpy",
    )

    first = data_process(data, validation_setting="B")
    second = data_process(data, validation_setting="B")

    first_split_sizes = tuple(len(split["y"]["data"]) for split in first[:3])
    second_split_sizes = tuple(len(split["y"]["data"]) for split in second[:3])
    assert first_split_sizes == second_split_sizes


@pytest.mark.parametrize(
    "ratio",
    [
        {"train": 0.7, "val": 0.1},
        {"train": 0.7, "val": 0.1, "test": 0.3},
        {"train": 0.8, "val": -0.1, "test": 0.3},
    ],
)
def test_validate_split_ratio_rejects_invalid_values(ratio):
    with pytest.raises((TypeError, ValueError)):
        validate_split_ratio(ratio)


def test_validate_split_ratio_returns_an_independent_copy():
    ratio = {"train": 0.7, "val": 0.1, "test": 0.2}

    validated = validate_split_ratio(ratio)
    validated["val"] = 0.5

    assert ratio["val"] == 0.1


@pytest.mark.parametrize(
    "interaction_data",
    [
        np.array([0, 1]),
        pd.DataFrame(columns=["instance_id", "target_id", "value"]),
        pd.DataFrame(
            {
                "instance_id": [0, 0],
                "target_id": [0, 0],
                "value": [0, 1],
            }
        ),
    ],
)
def test_process_interaction_data_rejects_ambiguous_boundaries(interaction_data):
    with pytest.raises(ValueError):
        process_interaction_data(interaction_data)


def test_process_interaction_data_keeps_an_independent_frame():
    source = pd.DataFrame(
        {
            "instance_id": [0, 1],
            "target_id": [0, 0],
            "value": [0, 1],
        }
    )

    processed = process_interaction_data(source)
    source.loc[0, "value"] = 99

    assert processed["data"]["value"].tolist() == [0, 1]


def test_interaction_format_check_is_silent_when_not_verbose(capsys):
    data = {
        split: {"y": {"original_format": "triplets"}}
        for split in ("train", "val", "test")
    }

    check_interaction_files_format(data, verbose=False)

    assert capsys.readouterr().out == ""


def test_variable_type_rejects_empty_values():
    with pytest.raises(ValueError, match="empty"):
        check_variable_type(pd.DataFrame(columns=["instance_id", "target_id", "value"]))


def test_triplet_novelty_detects_partially_new_ids():
    train = {
        "data": pd.DataFrame(
            {
                "instance_id": [0, 1],
                "target_id": [0, 1],
                "value": [0, 1],
            }
        ),
        "original_format": "triplets",
    }
    test = {
        "data": pd.DataFrame(
            {
                "instance_id": [1, 2],
                "target_id": [1, 2],
                "value": [1, 0],
            }
        ),
        "original_format": "triplets",
    }

    assert check_novel_instances(train, test) is True
    assert check_novel_targets(train, test) is True


def test_validation_setting_flags_require_booleans():
    with pytest.raises(TypeError, match="novel_instances_flag"):
        get_estimated_validation_setting(1, False, verbose=False)


@pytest.mark.parametrize(
    ("mode", "group_column"),
    [("macro", "target_id"), ("instance", "instance_id")],
)
def test_mtp_sampler_balances_within_requested_group(mode, group_column):
    data = pd.DataFrame(
        {
            "instance_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "target_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "value": [0, 0, 0, 1, 0, 1, 1, 1],
        }
    )
    if mode == "instance":
        data = pd.DataFrame(
            {
                "instance_id": [0, 0, 0, 0, 1, 1, 1, 1],
                "target_id": [0, 1, 2, 3, 0, 1, 2, 3],
                "value": [0, 0, 0, 1, 0, 1, 1, 1],
            }
        )

    sampler = MTPSampler(data.copy(), mode=mode)

    for _, group in sampler.data.groupby(group_column):
        assert group["value"].value_counts().nunique() == 1


def test_base_dataset_rejects_ids_outside_one_hot_dimensions():
    config = {
        "instance_branch_input_dim": 2,
        "target_branch_input_dim": 2,
        "use_instance_features": False,
        "use_target_features": False,
        "instance_branch_architecture": "MLP",
        "target_branch_architecture": "MLP",
    }
    interactions = {
        "data": pd.DataFrame(
            {
                "instance_id": [10],
                "target_id": [0],
                "value": [1],
            }
        )
    }

    with pytest.raises(ValueError, match="Instance IDs must be zero-based"):
        BaseDataset(config, interactions, None, None)
